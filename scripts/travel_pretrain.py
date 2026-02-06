"""
Continue pretraining d34 on travel blog posts from SQLite database.

Usage:
    python -m scripts.travel_pretrain --db-path=/path/to/blogs.db

This loads the d34 base model and continues pretraining on travel content
to give it better travel domain knowledge before SFT.

Near-stock nanochat setup: Muon + 8-bit Adam, bf16 weights,
gradient checkpointing. Optimized for 24GB GPUs (RTX 3090).
"""

import argparse
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import sqlite3
import random
import time
import gc
from contextlib import nullcontext

import torch
import bitsandbytes as bnb

from nanochat.gpt import GPT, GPTConfig
from nanochat.checkpoint_manager import load_model, save_checkpoint, find_last_step, load_checkpoint
from nanochat.common import get_base_dir, print0, autodetect_device_type
from nanochat.optim import polar_express_coeffs

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Continue pretraining on domain data from SQLite")
# Data
parser.add_argument("--db-path", type=str, required=True, help="path to SQLite database with text data")
parser.add_argument("--table-name", type=str, default="pages", help="table name")
parser.add_argument("--content-column", type=str, default="content", help="column with text content")
parser.add_argument("--title-column", type=str, default="title", help="column with title (prepended to content)")
# Model
parser.add_argument("--model-tag", type=str, default="d34", help="which base model to continue from")
parser.add_argument("--output-tag", type=str, default="d34-travel", help="tag for the output checkpoint")
# Training
parser.add_argument("--device-batch-size", type=int, default=8, help="batch size per step (8 fits on 24GB with bf16 + optimizer)")
parser.add_argument("--max-seq-len", type=int, default=2048, help="context length")
parser.add_argument("--total-batch-size", type=int, default=524288, help="total batch size in tokens")
parser.add_argument("--num-epochs", type=int, default=1, help="passes over the data")
parser.add_argument("--learning-rate-frac", type=float, default=0.1, help="fraction of original LR (10%% to avoid forgetting)")
parser.add_argument("--warmup-steps", type=int, default=100, help="LR warmup steps")
parser.add_argument("--save-every", type=int, default=1000, help="save checkpoint every N steps")
parser.add_argument("--keep-last-n", type=int, default=3, help="checkpoints to keep (0 = keep all)")
parser.add_argument("--resume", action="store_true", help="resume from latest checkpoint in output-tag directory")
# Eval
parser.add_argument("--eval-every", type=int, default=100, help="evaluate loss every N steps")
parser.add_argument("--eval-tokens", type=int, default=100000, help="tokens to evaluate on")
args = parser.parse_args()
# -----------------------------------------------------------------------------

device_type = autodetect_device_type()
device = torch.device(device_type)
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

# Load model and tokenizer
print0(f"Loading {args.model_tag} base model...")
model, tokenizer, meta = load_model('base', device, phase='train', model_tag=args.model_tag)
model.config.gradient_checkpointing = True
model = model.bfloat16()  # bf16 master weights: saves ~4 GB (8.6 -> 4.4 GB)
print0("Gradient checkpointing enabled, bf16 master weights")

gc.collect()
torch.cuda.empty_cache()

num_params = sum(p.numel() for p in model.parameters())
print0(f"Model parameters: {num_params:,}")

# Calculate gradient accumulation
tokens_per_batch = args.device_batch_size * args.max_seq_len
grad_accum_steps = args.total_batch_size // tokens_per_batch
print0(f"Batch size: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_batch:,} tokens")
print0(f"Total batch size: {args.total_batch_size:,} => grad accum steps: {grad_accum_steps}")

# -----------------------------------------------------------------------------
# Data loading from SQLite

class DomainDataset:
    def __init__(self, db_path, table_name, content_column, title_column, tokenizer, max_seq_len):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.content_column = content_column
        self.title_column = title_column
        self.table_name = table_name

        # Get total count
        self.cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        self.total_posts = self.cursor.fetchone()[0]
        print0(f"Found {self.total_posts:,} posts in database")

        # Get all URLs for shuffling
        self.cursor.execute(f"SELECT url FROM {table_name}")
        self.urls = [row[0] for row in self.cursor.fetchall()]

        # Token buffer for creating fixed-length sequences
        self.token_buffer = []
        self.post_idx = 0
        self.epoch = 0

    def shuffle(self):
        random.shuffle(self.urls)
        self.post_idx = 0

    def get_post_tokens(self, url):
        """Fetch and tokenize a single post."""
        self.cursor.execute(
            f"SELECT {self.title_column}, {self.content_column} FROM {self.table_name} WHERE url = ?",
            (url,)
        )
        row = self.cursor.fetchone()
        if row is None:
            return []

        title, content = row
        if not content:
            return []

        # Format: "Title\n\nContent"
        text = f"{title}\n\n{content}" if title else content

        # Tokenize with BOS token
        tokens = self.tokenizer(text, prepend="<|bos|>")
        return tokens

    def get_batch(self, batch_size):
        """Get a batch of (inputs, targets) for training."""
        sequences = []

        while len(sequences) < batch_size:
            # Fill buffer if needed
            while len(self.token_buffer) < self.max_seq_len + 1:
                if self.post_idx >= len(self.urls):
                    # End of epoch
                    self.epoch += 1
                    self.shuffle()
                    print0(f"Starting epoch {self.epoch}")

                tokens = self.get_post_tokens(self.urls[self.post_idx])
                self.post_idx += 1

                if tokens:
                    self.token_buffer.extend(tokens)

            # Extract a sequence from buffer
            seq = self.token_buffer[:self.max_seq_len + 1]
            self.token_buffer = self.token_buffer[self.max_seq_len:]
            sequences.append(seq)

        # Convert to tensors
        inputs = torch.tensor([s[:-1] for s in sequences], dtype=torch.long, device=device)
        targets = torch.tensor([s[1:] for s in sequences], dtype=torch.long, device=device)
        return inputs, targets

# -----------------------------------------------------------------------------
# Low-memory optimizer: per-parameter Muon + 8-bit Adam
# MuonAdamW stacks all same-shape params (~1.2 GB temporary tensors) which OOMs on 24GB.

class LowMemMuon(torch.optim.Optimizer):
    """Per-parameter Muon using Polar Express (from upstream nanochat.optim)."""
    def __init__(self, params, lr=0.02, momentum=0.95):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                g = p.grad
                buf.lerp_(g, 1 - momentum)
                g = g.lerp(buf, momentum)
                X = g.unsqueeze(0).bfloat16()
                X = X / (X.norm() * 1.02 + 1e-6)
                if X.size(-2) > X.size(-1):
                    for a, b, c in polar_express_coeffs[:5]:
                        A = X.mT @ X
                        B = b * A + c * (A @ A)
                        X = a * X + X @ B
                else:
                    for a, b, c in polar_express_coeffs[:5]:
                        A = X @ X.mT
                        B = b * A + c * (A @ A)
                        X = a * X + B @ X
                g = X.squeeze(0)
                scale = max(1.0, p.shape[-2] / p.shape[-1]) ** 0.5
                p.sub_(g, alpha=lr * scale)

def setup_optimizer(model, lr_frac):
    """Setup per-parameter optimizer with reduced LRs for continued pretraining."""
    orig = model._orig_mod if hasattr(model, '_orig_mod') else model
    model_dim = orig.config.n_embd
    dmodel_lr_scale = (model_dim / 768) ** -0.5

    matrix_params = list(orig.transformer.h.parameters())
    embedding_params = list(orig.transformer.wte.parameters())
    lm_head_params = list(orig.lm_head.parameters())
    scalar_params = [orig.resid_lambdas, orig.x0_lambdas]
    value_embeds_params = list(orig.value_embeds.parameters())

    adam8_groups = [
        dict(params=lm_head_params, lr=0.004 * dmodel_lr_scale * lr_frac),
        dict(params=embedding_params, lr=0.2 * dmodel_lr_scale * lr_frac),
    ]
    if value_embeds_params:
        adam8_groups.append(dict(params=value_embeds_params, lr=0.2 * dmodel_lr_scale * lr_frac))
    adam_optimizer = bnb.optim.Adam8bit(adam8_groups, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0)
    muon_optimizer = LowMemMuon(matrix_params, lr=0.02 * lr_frac, momentum=0.95)
    scalar_optimizer = torch.optim.Adam(scalar_params, lr=0.005 * lr_frac, betas=(0.8, 0.95))

    optimizers = [adam_optimizer, muon_optimizer, scalar_optimizer]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return optimizers

# -----------------------------------------------------------------------------
# Training loop

print0(f"\nLoading data from {args.db_path}...")
dataset = DomainDataset(args.db_path, args.table_name, args.content_column, args.title_column, tokenizer, args.max_seq_len)
dataset.shuffle()

# Estimate tokens (rough calculation to avoid memory overhead)
estimated_tokens = dataset.total_posts * 2200  # ~2200 tokens per post average
print0(f"Estimated total tokens: {estimated_tokens:,}")

num_steps_per_epoch = estimated_tokens // args.total_batch_size
total_steps = max(num_steps_per_epoch * args.num_epochs, 1)
print0(f"Estimated steps per epoch: {num_steps_per_epoch:,}")
print0(f"Total training steps: {total_steps:,}")

# Setup optimizer
gc.collect()
torch.cuda.empty_cache()
print0(f"\nSetting up optimizer with {args.learning_rate_frac:.1%} of original LR...")
optimizers = setup_optimizer(model, args.learning_rate_frac)

if torch.cuda.is_available():
    mem = torch.cuda.memory_allocated() / 1e9
    print0(f"Memory after setup: {mem:.2f} GB")

# LR schedule with warmup then constant
def get_lr_multiplier(step):
    if step < args.warmup_steps:
        return (step + 1) / args.warmup_steps
    return 1.0

# Training state
step = 0
tokens_seen = 0
smooth_loss = 0

base_dir = get_base_dir()
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", args.output_tag)

# Resume from checkpoint if requested
if args.resume:
    try:
        resume_step = find_last_step(checkpoint_dir)
        print0(f"Resuming from checkpoint at step {resume_step}...")
        model_data = torch.load(os.path.join(checkpoint_dir, f"model_{resume_step:06d}.pt"), map_location=device)
        model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
        model.load_state_dict(model_data, strict=True)
        del model_data
        gc.collect()
        step = resume_step
        tokens_seen = step * args.total_batch_size
        print0(f"Resumed from step {step}, tokens_seen={tokens_seen:,}")
    except FileNotFoundError:
        print0(f"No checkpoint found in {checkpoint_dir}, starting from scratch")

print0(f"\nStarting continued pretraining...")
print0(f"=" * 60)

start_time = time.time()

try:
    while dataset.epoch < args.num_epochs:
        # Evaluate periodically
        if step % args.eval_every == 0:
            model.eval()
            eval_losses = []
            eval_batches = min(5, args.eval_tokens // (args.device_batch_size * args.max_seq_len))
            with torch.no_grad():
                for _ in range(eval_batches):
                    x, y = dataset.get_batch(args.device_batch_size)
                    with autocast_ctx:
                        loss = model(x, y)
                    eval_losses.append(loss.item())
                    del x, y, loss
            eval_loss = sum(eval_losses) / len(eval_losses)
            print0(f"Step {step:6d} | eval loss: {eval_loss:.4f}")
            model.train()
            torch.cuda.empty_cache()

        # Save checkpoint
        if step > 0 and step % args.save_every == 0:
            model_config_kwargs = model.config.__dict__
            save_checkpoint(
                checkpoint_dir,
                step,
                model.state_dict(),
                None,
                {
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "source_model": args.model_tag,
                    "model_config": model_config_kwargs,
                },
                keep_last_n=args.keep_last_n,
            )

        # Training step with gradient accumulation
        t0 = time.time()
        for micro_step in range(grad_accum_steps):
            x, y = dataset.get_batch(args.device_batch_size)
            with autocast_ctx:
                loss = model(x, y)
            train_loss = loss.detach().item()
            loss = loss / grad_accum_steps
            loss.backward()
            tokens_seen += args.device_batch_size * args.max_seq_len
            del x, y, loss

        # Update LR
        lrm = get_lr_multiplier(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lrm

        # Optimizer step
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        dt = time.time() - t0

        # Logging
        smooth_loss = 0.9 * smooth_loss + 0.1 * train_loss
        tok_per_sec = args.total_batch_size / dt

        if step % 10 == 0:
            elapsed = time.time() - start_time
            print0(f"Step {step:6d} | loss: {smooth_loss:.4f} | lr: {lrm:.3f} | tok/s: {tok_per_sec:,.0f} | epoch: {dataset.epoch} | time: {elapsed/60:.1f}m")

        step += 1

except KeyboardInterrupt:
    print0("\nInterrupted by user")

# Final save
print0(f"\nSaving final checkpoint...")
model_config_kwargs = model.config.__dict__
save_checkpoint(
    checkpoint_dir,
    step,
    model.state_dict(),
    None,
    {
        "step": step,
        "tokens_seen": tokens_seen,
        "source_model": args.model_tag,
        "model_config": model_config_kwargs,
        "final": True,
    },
    keep_last_n=args.keep_last_n,
)

total_time = time.time() - start_time
print0(f"\nTraining complete!")
print0(f"Total steps: {step:,}")
print0(f"Total tokens: {tokens_seen:,}")
print0(f"Total time: {total_time/3600:.1f} hours")
print0(f"Checkpoint saved to: {checkpoint_dir}")
