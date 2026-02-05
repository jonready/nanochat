"""
Memory-efficient SFT for 24GB GPUs.
Near-stock nanochat: MuonAdamW, bf16 weights, gradient checkpointing.

Usage:
    python -m scripts.chat_sft_lowmem --source=base --model-tag=d34-travel
"""

import argparse
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gc
import time
from contextlib import nullcontext

from nanochat.common import get_base_dir, print0, autodetect_device_type
from nanochat.checkpoint_manager import load_model, save_checkpoint, find_last_step
from nanochat.engine import Engine
from scripts.chat_eval import run_chat_eval

from tasks.common import TaskMixture
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Memory-efficient SFT for 24GB GPUs")
# Model loading
parser.add_argument("--source", type=str, default="base", help="base|sft - which checkpoint to load from")
parser.add_argument("--model-tag", type=str, default="d34-travel", help="model tag to load")
parser.add_argument("--model-step", type=int, default=None, help="step to load from (None = latest)")
# Training
parser.add_argument("--device-batch-size", type=int, default=4, help="fits on 24GB with bf16 + grad checkpointing")
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs")
parser.add_argument("--target-examples-per-step", type=int, default=32, help="grad accum to reach this")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="stock unembedding LR")
parser.add_argument("--embedding-lr", type=float, default=0.2, help="stock embedding LR")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="stock matrix LR")
parser.add_argument("--init-lr-frac", type=float, default=0.02, help="initial LR fraction (stock default)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay")
# Eval
parser.add_argument("--eval-every", type=int, default=100, help="evaluate val loss every N steps")
parser.add_argument("--eval-steps", type=int, default=100, help="number of eval batches")
parser.add_argument("--eval-metrics-every", type=int, default=200, help="evaluate accuracy metrics every N steps")
parser.add_argument("--eval-metrics-max-problems", type=int, default=1024, help="max problems for metrics eval")
parser.add_argument("--save-every", type=int, default=200, help="save checkpoint every N steps")
parser.add_argument("--keep-last-n", type=int, default=3, help="checkpoints to keep (0 = keep all)")
parser.add_argument("--resume", action="store_true", help="resume from latest checkpoint")
args = parser.parse_args()
# -----------------------------------------------------------------------------

device_type = autodetect_device_type()
device = torch.device(device_type)
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

base_dir = get_base_dir()

# Load model
start_step = 0
if args.resume:
    print0(f"Loading model config from {args.source}/{args.model_tag}...")
    model, tokenizer, meta = load_model(args.source, device, phase="train", model_tag=args.model_tag, step=args.model_step)
    depth = model.config.n_layer
    output_tag = f"d{depth}-travel-sft"
    checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_tag)

    try:
        resume_step = find_last_step(checkpoint_dir)
        print0(f"Found SFT checkpoint at step {resume_step}, loading...")
        model_path = os.path.join(checkpoint_dir, f"model_{resume_step:06d}.pt")
        model_data = torch.load(model_path, map_location="cpu")
        model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
        model.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()
        model.load_state_dict(model_data, strict=True)
        del model_data
        gc.collect()
        model.to(device)
        start_step = resume_step + 1
        print0(f"Resumed from step {resume_step}! Starting from step {start_step}")
    except FileNotFoundError:
        print0(f"No checkpoint found in {checkpoint_dir}, starting from scratch")
else:
    print0(f"Loading model from {args.source}/{args.model_tag}...")
    model, tokenizer, meta = load_model(args.source, device, phase="train", model_tag=args.model_tag, step=args.model_step)

model.config.gradient_checkpointing = True
model = model.bfloat16()  # bf16 master weights: saves ~4 GB
print0("Gradient checkpointing enabled, bf16 master weights")

orig_model = model
engine = Engine(model, tokenizer)

num_params = sum(p.numel() for p in model.parameters())
print0(f"Model parameters: {num_params:,}")

# -----------------------------------------------------------------------------
# Task data mixture - match stock chat_sft.py
identity_conversations_filepath = os.path.join(get_base_dir(), "identity_conversations.jsonl")
train_ds = TaskMixture([
    ARC(subset="ARC-Easy", split="train"),       # 2.3K rows
    ARC(subset="ARC-Challenge", split="train"),   # 1.1K rows
    GSM8K(subset="main", split="train"),          # 8K rows
    SmolTalk(split="train", stop=10_000),         # 10K rows (match stock)
    CustomJSON(filepath=identity_conversations_filepath),  # 1K rows
    SimpleSpelling(size=300, split="train"),       # 300 rows (match stock)
    SpellingBee(size=300, split="train"),          # 300 rows (match stock)
])
val_ds = SmolTalk(split="test")
print0(f"Training examples: {len(train_ds):,}")

# -----------------------------------------------------------------------------
# DataLoader

def sft_data_generator(dataset, batch_size):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")

    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, mask in batch) - 1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n-1] = ids_tensor[:-1]
            row_targets = ids_tensor[1:]
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1
            targets[i, :n-1] = row_targets
        return inputs.to(device), targets.to(device)

    batch = []
    while True:
        for i in range(len(dataset)):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []

examples_per_step = args.device_batch_size
assert args.target_examples_per_step % examples_per_step == 0, \
    f"target_examples_per_step ({args.target_examples_per_step}) must be divisible by device_batch_size ({args.device_batch_size})"
grad_accum_steps = args.target_examples_per_step // examples_per_step
print0(f"Target examples per step: {args.target_examples_per_step}")
print0(f"Device batch size: {args.device_batch_size}")
print0(f"Grad accum steps: {grad_accum_steps}")

num_iterations = (len(train_ds) // args.target_examples_per_step) * args.num_epochs
train_loader = sft_data_generator(train_ds, batch_size=args.device_batch_size)
build_val_loader = lambda: sft_data_generator(val_ds, batch_size=args.device_batch_size)

print0(f"Number of iterations: {num_iterations:,}")

# -----------------------------------------------------------------------------
# Setup optimizer using model.setup_optimizer() (uses MuonAdamW)
# With bf16 model, optimizer states are also bf16, preserving memory savings.

gc.collect()
torch.cuda.empty_cache()

optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)

# Apply init_lr_frac (stock: start at 2% of base LR)
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

print0(f"Initial LRs (at init_lr_frac={args.init_lr_frac}):")
for group in optimizer.param_groups:
    print0(f"  {group['kind']}: {group['lr']:.6f}")

if torch.cuda.is_available():
    mem = torch.cuda.memory_allocated() / 1e9
    print0(f"Memory after setup: {mem:.2f} GB")

# -----------------------------------------------------------------------------
# Training loop

def get_lr_multiplier(it):
    return 1.0 - it / num_iterations

# Set checkpoint_dir if not already set
if not args.resume:
    depth = model.config.n_layer
    output_tag = f"d{depth}-travel-sft"
    checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_tag)

model.train()
smooth_loss = 0
start_time = time.time()

print0(f"\nStarting SFT training...")
print0("=" * 60)

step = start_step
for step in range(start_step, num_iterations):
    last_step = step == num_iterations - 1

    # Evaluate validation loss
    if last_step or step % args.eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        losses = []
        with torch.no_grad():
            for _ in range(args.eval_steps):
                val_inputs, val_targets = next(val_loader)
                with autocast_ctx:
                    loss = model(val_inputs, val_targets)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    losses.append(loss.item())
                del val_inputs, val_targets, loss
        val_loss = sum(losses) / len(losses) if losses else float('nan')
        print0(f"Step {step:05d} | val_loss: {val_loss:.4f}")
        model.train()
        torch.cuda.empty_cache()

    # Evaluate accuracy metrics
    if last_step or (step > 0 and step % args.eval_metrics_every == 0):
        model.eval()
        metrics = {}
        with torch.no_grad(), autocast_ctx:
            metrics["mmlu_acc"] = run_chat_eval("MMLU", model, tokenizer, engine,
                batch_size=args.device_batch_size*2, max_problems=args.eval_metrics_max_problems)
            metrics["arc_easy_acc"] = run_chat_eval("ARC-Easy", model, tokenizer, engine,
                batch_size=args.device_batch_size*2, max_problems=args.eval_metrics_max_problems)
        metrics_str = ', '.join(f'{k}: {v:.4f}' for k, v in metrics.items())
        print0(f"Step {step:05d} | {metrics_str}")
        model.train()

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
                "source_model": args.model_tag,
                "model_config": model_config_kwargs,
            },
            keep_last_n=args.keep_last_n,
        )

    if last_step:
        break

    # Training step with gradient accumulation
    num_tokens = 0
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(train_loader)
        with autocast_ctx:
            loss = model(train_inputs, train_targets)
        train_loss = loss.detach().item()
        loss = loss / grad_accum_steps
        loss.backward()
        num_tokens += (train_targets >= 0).sum().item()
        del train_inputs, train_targets, loss

    # LR schedule (linear decay, matching stock)
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm

    # Optimizer step
    optimizer.step()
    model.zero_grad(set_to_none=True)

    # Logging
    smooth_loss = 0.95 * smooth_loss + 0.05 * train_loss if smooth_loss > 0 else train_loss
    if step % 10 == 0:
        elapsed = time.time() - start_time
        print0(f"Step {step:05d}/{num_iterations} | loss: {smooth_loss:.4f} | lrm: {lrm:.3f} | num_tokens: {num_tokens:,} | {elapsed/60:.1f}m elapsed")

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
        "source_model": args.model_tag,
        "val_loss": val_loss,
        "model_config": model_config_kwargs,
        "final": True,
    },
    keep_last_n=args.keep_last_n,
)

print0(f"\nSFT complete!")
print0(f"Checkpoint saved to: {checkpoint_dir}")
