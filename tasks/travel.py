"""
Travel knowledge evaluation.
Hand-curated multiple-choice questions testing geographic, cultural,
and practical travel knowledge — the kind of information densely
represented in travel blog corpora.
"""

import random

from tasks.common import Task, render_mc


# Each entry: (question, choices_list, correct_index)
# Correct index is 0-based into the choices list.
# Convention: correct answer is always listed FIRST (index 0) for readability.
# The TravelKnowledge class shuffles choices deterministically at init time
# so the correct answer is evenly distributed across A/B/C/D during eval.
TRAVEL_QUESTIONS = [
    # ── Dolomites & Italian Alps ──────────────────────────────────────────
    (
        "The Tre Cime di Lavaredo, one of the most iconic hikes in the Dolomites, is located in which Italian region?",
        ["Trentino-Alto Adige", "Lombardy", "Veneto", "Friuli Venezia Giulia"],
        0,
    ),
    (
        "What is the name of the famous multi-day hiking route that loops through the Dolomites, typically completed in 8-13 days?",
        ["Alta Via 1", "Tour du Mont Blanc", "Walker's Haute Route", "GR20"],
        0,
    ),
    (
        "The Seceda ridgeline, famous for its dramatic views of the Odle/Geisler mountain group, is accessed from which valley?",
        ["Val Gardena", "Val di Fassa", "Val Pusteria", "Val di Fiemme"],
        0,
    ),
    (
        "Which lake in the Dolomites is known for its striking turquoise color and sits at the base of the Croda da Lago?",
        ["Lago di Braies", "Lago di Sorapis", "Lago di Carezza", "Lago di Misurina"],
        1,
    ),
    (
        "Rifugios in the Dolomites are mountain huts that typically offer which services to hikers?",
        ["Meals and overnight accommodation", "Only emergency shelter", "Guided climbing only", "Equipment rental only"],
        0,
    ),

    # ── Southeast Asia ────────────────────────────────────────────────────
    (
        "Railay Beach in Krabi, Thailand is primarily accessed by which mode of transportation?",
        ["Longtail boat", "Road bridge", "Cable car", "Ferry terminal"],
        0,
    ),
    (
        "Which Vietnamese city is the typical starting point for boat trips to Ha Long Bay?",
        ["Hanoi", "Ho Chi Minh City", "Da Nang", "Hue"],
        0,
    ),
    (
        "Angkor Wat, the largest religious monument in the world, is located near which Cambodian city?",
        ["Siem Reap", "Phnom Penh", "Battambang", "Sihanoukville"],
        0,
    ),
    (
        "The island of Bali belongs to which country?",
        ["Indonesia", "Philippines", "Thailand", "Malaysia"],
        0,
    ),
    (
        "Luang Prabang, a UNESCO World Heritage city known for its morning alms-giving ceremony, is in which country?",
        ["Laos", "Myanmar", "Cambodia", "Vietnam"],
        0,
    ),

    # ── Europe ────────────────────────────────────────────────────────────
    (
        "The Camino de Santiago pilgrimage route traditionally ends in which Spanish city?",
        ["Santiago de Compostela", "Barcelona", "Madrid", "Seville"],
        0,
    ),
    (
        "Plitvice Lakes National Park, famous for its cascading turquoise lakes, is located in which country?",
        ["Croatia", "Slovenia", "Montenegro", "Bosnia and Herzegovina"],
        0,
    ),
    (
        "The Cinque Terre in Italy consists of how many coastal villages?",
        ["Five", "Three", "Seven", "Four"],
        0,
    ),
    (
        "Hallstatt, the lakeside village often called the most photographed village in Europe, is in which country?",
        ["Austria", "Switzerland", "Germany", "Czech Republic"],
        0,
    ),
    (
        "The Ring Road (Route 1) that circles the entire island is the main highway of which country?",
        ["Iceland", "Ireland", "Sardinia", "Sicily"],
        0,
    ),
    (
        "Dubrovnik's Old Town, a UNESCO site famous for its medieval walls, served as a filming location for which TV series?",
        ["Game of Thrones", "The Witcher", "Vikings", "Outlander"],
        0,
    ),
    (
        "Budapest is divided into Buda and Pest by which river?",
        ["Danube", "Vltava", "Tisza", "Rhine"],
        0,
    ),

    # ── Central & South America ───────────────────────────────────────────
    (
        "Machu Picchu is most commonly reached by hiking which famous trail?",
        ["Inca Trail", "Huayhuash Circuit", "Lares Trek", "Ausangate Trek"],
        0,
    ),
    (
        "The Salar de Uyuni, the world's largest salt flat, is located in which country?",
        ["Bolivia", "Chile", "Peru", "Argentina"],
        0,
    ),
    (
        "Cenotes, the natural sinkholes popular for swimming and diving, are most commonly found in which region?",
        ["Yucatan Peninsula, Mexico", "Costa Rican highlands", "Patagonia, Argentina", "Amazon Basin, Brazil"],
        0,
    ),

    # ── Africa & Middle East ──────────────────────────────────────────────
    (
        "A safari in the Serengeti National Park to witness the Great Migration would take you to which country?",
        ["Tanzania", "Kenya", "South Africa", "Botswana"],
        0,
    ),
    (
        "The ancient city of Petra, carved into rose-red cliffs, is in which country?",
        ["Jordan", "Egypt", "Lebanon", "Turkey"],
        0,
    ),
    (
        "Table Mountain, one of the New7Wonders of Nature, overlooks which city?",
        ["Cape Town", "Johannesburg", "Nairobi", "Windhoek"],
        0,
    ),

    # ── Oceania ───────────────────────────────────────────────────────────
    (
        "The Milford Track, often called 'the finest walk in the world', is located in which country?",
        ["New Zealand", "Australia", "Tasmania", "Fiji"],
        0,
    ),
    (
        "The Great Barrier Reef, the world's largest coral reef system, lies off the coast of which Australian state?",
        ["Queensland", "New South Wales", "Western Australia", "Victoria"],
        0,
    ),

    # ── Japan & East Asia ─────────────────────────────────────────────────
    (
        "The Fushimi Inari Shrine, known for thousands of vermillion torii gates, is in which Japanese city?",
        ["Kyoto", "Tokyo", "Osaka", "Nara"],
        0,
    ),
    (
        "What is the name of the bullet train system in Japan?",
        ["Shinkansen", "Maglev", "KTX", "Tokaido"],
        0,
    ),
    (
        "Myeongdong, a major shopping and street food district, is located in which Asian capital?",
        ["Seoul", "Tokyo", "Taipei", "Beijing"],
        0,
    ),

    # ── Practical travel knowledge ────────────────────────────────────────
    (
        "What does the travel term 'red-eye flight' refer to?",
        ["An overnight flight arriving early morning", "A flight with multiple stopovers", "A discounted last-minute fare", "A flight during peak holiday season"],
        0,
    ),
    (
        "In most countries, what does a blue flag on a beach indicate?",
        ["The beach meets high environmental and water quality standards", "The beach is closed for maintenance", "Swimming is prohibited due to currents", "The beach is privately owned"],
        0,
    ),
    (
        "What is the main advantage of booking 'open-jaw' flights?",
        ["Flying into one city and out of another to avoid backtracking", "Getting a free stopover at a hub city", "Avoiding baggage fees on connecting flights", "Locking in a fixed exchange rate for the ticket price"],
        0,
    ),

    # ── North America ─────────────────────────────────────────────────────
    (
        "The Pacific Crest Trail (PCT) spans from Mexico to Canada through which US states?",
        ["California, Oregon, and Washington", "California, Nevada, and Oregon", "Arizona, California, and Oregon", "California, Oregon, and Idaho"],
        0,
    ),
    (
        "Which US national park is famous for its hoodoo rock formations and is located in Utah?",
        ["Bryce Canyon", "Arches", "Capitol Reef", "Canyonlands"],
        0,
    ),
    (
        "The town of Banff, a gateway to the Canadian Rockies, is in which province?",
        ["Alberta", "British Columbia", "Saskatchewan", "Manitoba"],
        0,
    ),

    # ── South & Central Asia ──────────────────────────────────────────────
    (
        "Everest Base Camp trek begins in which country?",
        ["Nepal", "Tibet (China)", "India", "Bhutan"],
        0,
    ),
    (
        "The Golden Triangle in Southeast Asia refers to the border area of Thailand, Laos, and which third country?",
        ["Myanmar", "Vietnam", "Cambodia", "China"],
        0,
    ),
    (
        "Sri Lanka's Cultural Triangle includes Sigiriya Rock Fortress, the ancient cities of Anuradhapura and which other city?",
        ["Polonnaruwa", "Kandy", "Galle", "Dambulla"],
        0,
    ),

    # ── Islands & beaches ─────────────────────────────────────────────────
    (
        "The Maldives is an island nation in which ocean?",
        ["Indian Ocean", "Pacific Ocean", "Atlantic Ocean", "Arabian Sea"],
        0,
    ),
    (
        "Santorini, known for its white-washed buildings with blue domes, is part of which country?",
        ["Greece", "Turkey", "Italy", "Croatia"],
        0,
    ),
    (
        "The Galapagos Islands, famous for their unique wildlife, belong to which country?",
        ["Ecuador", "Colombia", "Peru", "Chile"],
        0,
    ),

    # ── Food & culture travel ─────────────────────────────────────────────
    (
        "Pad Thai, Tom Yum soup, and green curry are signature dishes of which country's cuisine?",
        ["Thailand", "Vietnam", "Indonesia", "Malaysia"],
        0,
    ),
    (
        "A traditional Moroccan tagine is named after what?",
        ["The cone-shaped clay pot it is cooked in", "The spice blend used in the dish", "The region where it originated", "The chef who invented it"],
        0,
    ),
    (
        "The Tsukiji Outer Market, famous for fresh sushi and street food, is in which city?",
        ["Tokyo", "Osaka", "Kyoto", "Sapporo"],
        0,
    ),

    # ── More Dolomites / Alps specifics ───────────────────────────────────
    (
        "Which pass connects the Val Gardena and Val Badia valleys and is a popular cycling climb in the Dolomites?",
        ["Passo Gardena", "Passo Sella", "Passo Pordoi", "Passo Falzarego"],
        0,
    ),
    (
        "The Marmolada, the highest peak in the Dolomites, reaches approximately what elevation?",
        ["3,343 meters", "2,999 meters", "4,061 meters", "3,798 meters"],
        0,
    ),
    (
        "Which UNESCO-listed area in northern Italy encompasses the Dolomites mountain range?",
        ["The Dolomites UNESCO World Heritage Site", "The Italian Alps Biosphere Reserve", "The Tyrol Natural Heritage Zone", "The Alpine Geological Park"],
        0,
    ),

    # ── Trekking & outdoor ────────────────────────────────────────────────
    (
        "The W Trek, a popular multi-day hike, is located in Torres del Paine National Park in which country?",
        ["Chile", "Argentina", "Peru", "Bolivia"],
        0,
    ),
    (
        "What is the typical altitude where hikers may first experience symptoms of altitude sickness?",
        ["Above 2,500 meters (8,000 feet)", "Above 1,000 meters (3,300 feet)", "Above 5,000 meters (16,400 feet)", "Above 500 meters (1,600 feet)"],
        0,
    ),
    (
        "The Tongariro Alpine Crossing, one of New Zealand's Great Walks, passes through a volcanic landscape on which island?",
        ["North Island", "South Island", "Stewart Island", "Waiheke Island"],
        0,
    ),
    (
        "The O Trek in Torres del Paine must be hiked in which direction?",
        ["Counterclockwise only", "Clockwise only", "Either direction", "It varies by season"],
        0,
    ),

    # ── Transportation & logistics ────────────────────────────────────────
    (
        "The Eurail Pass is valid for train travel in how many European countries (approximately)?",
        ["33", "15", "50", "8"],
        0,
    ),
    (
        "Which airline alliance includes members such as Lufthansa, United Airlines, and Air Canada?",
        ["Star Alliance", "Oneworld", "SkyTeam", "Open Skies"],
        0,
    ),

    # ── Insider / blog-level knowledge ────────────────────────────────────
    # These test the kind of practical details that travel bloggers write about.
    (
        "When visiting Lago di Braies (Pragser Wildsee) in the Dolomites during peak summer, what is the main logistical challenge?",
        ["Parking fills up very early and access is restricted", "The lake is closed for swimming", "The trail is only open to guided groups", "Entry requires a multi-day hiking permit"],
        0,
    ),
    (
        "On the Inca Trail to Machu Picchu, what is the name of the highest pass hikers must cross?",
        ["Dead Woman's Pass (Warmiwanusqa)", "Sun Gate (Intipunku)", "Runkurakay Pass", "Salkantay Pass"],
        0,
    ),
    (
        "In Iceland, the Golden Circle tourist route includes Thingvellir National Park, the Geysir geothermal area, and which waterfall?",
        ["Gullfoss", "Skogafoss", "Seljalandsfoss", "Dettifoss"],
        0,
    ),
    (
        "What is the main reason Patagonia's hiking season runs from November to March?",
        ["Those months are summer in the Southern Hemisphere", "The trails are closed for wildlife breeding the rest of the year", "Snow blocks all passes from April to October", "Park fees are waived during those months"],
        0,
    ),
    (
        "When taking the ferry from Sorrento to Capri or Ischia, departures leave from which type of port area?",
        ["Marina Piccola (the small harbor)", "The main cruise terminal", "A beach pier near the town center", "Naples central port only"],
        0,
    ),
    (
        "The Kungsleden (King's Trail) is a famous long-distance hiking path in which Scandinavian country?",
        ["Sweden", "Norway", "Finland", "Denmark"],
        0,
    ),
    (
        "What is the most common way to travel between islands in the Philippines?",
        ["Domestic flights and ferries", "Bridges and highways", "Helicopter transfers", "Ziplines between islands"],
        0,
    ),
    (
        "Overtourism at Maya Bay in Thailand (made famous by the movie 'The Beach') led authorities to do what?",
        ["Close the bay temporarily to allow ecosystem recovery", "Build an artificial reef nearby as an alternative", "Limit visits to scuba divers only", "Charge a premium entry fee of $100 per person"],
        0,
    ),
    (
        "In Japan, what is an onsen?",
        ["A natural hot spring bath", "A capsule hotel", "A traditional tea ceremony", "A bullet train station lounge"],
        0,
    ),
    (
        "The Via Ferrata routes in the Dolomites were originally built during which conflict?",
        ["World War I", "World War II", "The Napoleonic Wars", "The Italian Wars of Independence"],
        0,
    ),
    (
        "What distinguishes the GR20 trail in Corsica from most other European long-distance hikes?",
        ["It is considered one of the most difficult GR trails in Europe", "It follows a flat coastal route", "It can only be hiked with a licensed guide", "It is entirely above the tree line"],
        0,
    ),
    (
        "Chefchaouen, the city famous for its blue-painted buildings, is located in which country?",
        ["Morocco", "Greece", "India", "Portugal"],
        0,
    ),
    (
        "What is the name of the overnight train between Bangkok and Chiang Mai that is popular with budget travelers?",
        ["The sleeper train (Thai Railways second-class sleeper)", "The Orient Express of Asia", "The Mekong Express", "The Thai Bullet"],
        0,
    ),
    (
        "When trekking to Everest Base Camp, most hikers fly into which small mountain airstrip known for its short runway?",
        ["Lukla (Tenzing-Hillary Airport)", "Kathmandu (Tribhuvan Airport)", "Pokhara Airport", "Jomsom Airport"],
        0,
    ),
    (
        "The Amalfi Coast road (SS163) in Italy is famous among travelers for what characteristic?",
        ["Narrow, winding cliffside roads with dramatic sea views", "Being the longest coastal highway in Europe", "Having no speed limits along its length", "Being accessible only by electric vehicles"],
        0,
    ),
    (
        "What is a 'digital nomad visa' designed for?",
        ["Remote workers who want to live in a foreign country while working for employers elsewhere", "Tech workers relocating permanently", "Short-term tourist stays under 30 days", "Students attending coding bootcamps abroad"],
        0,
    ),

    # ── Gotcha questions: Patagonia & South America ─────────────────────────
    # These test counterintuitive, blog-level knowledge.
    (
        "Unlike most glaciers worldwide, Perito Moreno Glacier in Patagonia is notable because it:",
        ["Is roughly stable or advancing, unlike most retreating glaciers", "Is retreating faster than any other glacier in South America", "Has completely stopped calving due to recent temperature changes", "Is the only glacier in Patagonia that feeds into the Pacific Ocean"],
        0,
    ),
    (
        "To hike the W Trek in Torres del Paine, accommodation (refugios and campsites) must be:",
        ["Reserved in advance through two separate private companies that operate different sections", "Arranged upon arrival at each refugio on a first-come, first-served basis", "Reserved through a single centralized park booking system run by CONAF", "Booked through the Chilean government tourism website exclusively"],
        0,
    ),
    (
        "On the W Trek, to see the actual three granite 'Torres' (towers) that give the park its name, you must:",
        ["Hike up the steep Valle Ascencio to the Mirador Base Las Torres", "Look up from any point along the main trail — they are visible throughout", "Hike to the Mirador Frances in the French Valley", "Take a boat across Grey Lake to a viewing platform"],
        0,
    ),
    (
        "To hike to the base of Fitz Roy (Laguna de los Tres) from El Chalten, Argentina, what entrance fee or permit is required?",
        ["No fee or permit — the trail is free and open access", "A national park fee of approximately $35 USD payable at the trailhead", "A free permit that must be obtained online in advance", "A fee included in the El Chalten tourist tax"],
        0,
    ),
    (
        "Rainbow Mountain (Vinicunca) in Peru, commonly sold as a day trip from Cusco, sits at what approximate elevation?",
        ["5,200 meters (17,060 ft) — comparable to Everest Base Camp", "3,800 meters (12,500 ft) — similar to Cusco itself", "4,200 meters (13,780 ft) — similar to La Paz, Bolivia", "2,400 meters (7,870 ft) — similar to Mexico City"],
        0,
    ),
    (
        "The Inca Trail to Machu Picchu has a daily limit of 500 people. How many of those spots are typically available for actual hikers?",
        ["About 200, because guides and porters count toward the 500 limit", "All 500, since guides and porters are counted separately", "About 400, with 100 reserved for park staff", "About 50, because most spots go to Peruvian citizens first"],
        0,
    ),
    (
        "What is the most common way visitors reach the entrance to Machu Picchu?",
        ["Train to Aguas Calientes followed by a bus up the mountain", "Hiking the 4-day Inca Trail", "Driving a rental car to a parking lot at the entrance", "Helicopter from Cusco directly to the site"],
        0,
    ),

    # ── Gotcha questions: Southeast Asia ─────────────────────────────────────
    (
        "Angkor Wat is unusual among temples in the Angkor complex because it faces which direction?",
        ["West", "East", "North", "South"],
        0,
    ),
    (
        "What major change affected the visitor experience at Bagan's ancient temples in Myanmar after 2016?",
        ["Climbing most temples was banned after earthquake damage and conservation concerns", "Entry fees were raised to over $200 per day", "All temples were closed for restoration until 2025", "Visitors were required to hire government-approved guides for all visits"],
        0,
    ),
    (
        "During Bali's Nyepi (Day of Silence), which of the following occurs?",
        ["The international airport closes and tourists must stay in their hotels", "Only Indonesian citizens are allowed on the streets", "Temples hold special open-door ceremonies for visitors", "Beach areas remain open but inland roads are restricted"],
        0,
    ),
    (
        "What surprises many travelers flying from Kuala Lumpur to Kota Kinabalu (Sabah) in Malaysian Borneo?",
        ["They must pass through immigration control despite both being in Malaysia", "The flight requires a transit stop in Singapore due to airspace restrictions", "Malaysian ringgit is not accepted in Sabah, which uses its own currency", "Flights are operated exclusively by Borneo-based airlines"],
        0,
    ),
    (
        "The Mekong slow boat from Huay Xai (Thai-Lao border) to Luang Prabang typically takes how long?",
        ["Two days with an overnight stop in Pak Beng", "About 4-5 hours with no stops", "One full day from dawn to dusk", "Three to four days with stops in multiple villages"],
        0,
    ),
    (
        "What is the most commonly reported tourist scam near Bangkok's Grand Palace?",
        ["Tuk-tuk drivers telling tourists the palace is closed and redirecting them to gem shops", "Vendors selling counterfeit tickets at double the price", "Pickpockets posing as monks asking for donations", "Tour guides charging for entry to areas that are actually free"],
        0,
    ),

    # ── Gotcha questions: Europe ─────────────────────────────────────────────
    (
        "To receive the Compostela certificate upon completing the Camino de Santiago, what is the minimum distance a pilgrim must walk?",
        ["The last 100 kilometers", "The full route from St-Jean-Pied-de-Port (about 800 km)", "The last 200 kilometers", "Any distance, as long as you carry a credential"],
        0,
    ),
    (
        "To hike the Sentiero Azzurro (Blue Trail) connecting the five villages of Cinque Terre, what do hikers need?",
        ["A Cinque Terre Card (paid hiking permit)", "Nothing — the trail is free and open to all", "A guided tour booking through the national park", "An Italian national park annual membership"],
        0,
    ),
    (
        "The Tour du Mont Blanc hiking circuit passes through which countries?",
        ["France, Italy, and Switzerland", "France only, circling the French side of Mont Blanc", "France and Italy", "France, Italy, Switzerland, and Austria"],
        0,
    ),
    (
        "Trolltunga, Norway's famous cliff-edge rock formation, requires what to visit during summer?",
        ["A round-trip hike of approximately 27 km taking 10-12 hours", "A short 20-minute walk from the parking area", "A cable car ride followed by a 1 km path", "A guided boat tour across the fjord below"],
        0,
    ),
    (
        "Unlike England and Wales, Scotland's Land Reform Act (2003) grants hikers which unusual right?",
        ["The legal right to wild camp on most land", "Free entry to all castles and historic sites", "Free use of any mountain bothy for up to one week", "The right to fish in any river without a permit"],
        0,
    ),
    (
        "At the mountain huts along Sweden's Kungsleden trail, what meal service is typically available?",
        ["Hikers must cook their own food using the hut's kitchen facilities", "Full breakfast and dinner are included in the overnight fee", "A restaurant serves traditional Swedish meals", "All meals are provided but must be pre-ordered online"],
        0,
    ),

    # ── Gotcha questions: Nepal, Himalaya & Kilimanjaro ──────────────────────
    (
        "When trekking to Everest Base Camp, where do most hikers go for the classic panoramic view of Everest's summit?",
        ["Kala Patthar (5,644m) — Everest's summit is not visible from Base Camp itself", "Everest Base Camp itself (5,364m)", "Namche Bazaar viewpoint (3,440m)", "Tengboche Monastery (3,867m)"],
        0,
    ),
    (
        "The Annapurna Circuit in Nepal is almost always hiked counterclockwise. What is the primary reason?",
        ["Gradual altitude gain allows proper acclimatization before Thorong La Pass", "The trail markers only face counterclockwise", "Teahouses are only open to counterclockwise hikers", "It is legally required by Nepali park regulations"],
        0,
    ),
    (
        "Thorong La Pass, the highest point on the Annapurna Circuit, sits at 5,416 meters. How does this compare to Everest Base Camp?",
        ["It is higher than Everest Base Camp (5,364m)", "It is about 1,000 meters lower than Everest Base Camp", "It is exactly the same elevation as Everest Base Camp", "It is about 500 meters higher than Everest Base Camp"],
        0,
    ),
    (
        "What is the primary challenge that prevents climbers from reaching the summit of Mount Kilimanjaro?",
        ["Altitude sickness from the 5,895m elevation — no technical climbing is required", "Technical rock climbing on the summit ridge", "Crevasse crossings on the glacier", "Multi-day exposure to sub-zero blizzard conditions"],
        0,
    ),
    (
        "On Mount Kilimanjaro, the Marangu Route ('Coca-Cola Route') is the most popular. How does its summit success rate compare to other routes?",
        ["It has one of the lowest success rates due to insufficient acclimatization time", "It has the highest success rate because hut accommodation means hikers sleep better", "All routes have identical success rates of about 85%", "It has a moderate success rate, ranking third out of six routes"],
        0,
    ),
    (
        "What is required to trek the Manaslu Circuit in Nepal that is NOT required for the Annapurna Circuit or Everest Base Camp?",
        ["A licensed guide is legally mandatory, and a restricted area permit is needed", "Supplemental oxygen must be carried above 4,000 meters", "Trekkers must complete a government-administered fitness test", "A helicopter evacuation deposit must be paid in advance"],
        0,
    ),

    # ── Gotcha questions: Japan ──────────────────────────────────────────────
    (
        "A tourist with a Japan Rail Pass wants to take the Tokaido Shinkansen from Tokyo to Kyoto. Which train can they ride?",
        ["Hikari (the second-fastest service)", "Nozomi (the fastest, most frequent service)", "Any Shinkansen — the JR Pass covers all services", "Only local (non-Shinkansen) JR trains"],
        0,
    ),
    (
        "A tourist plans to see cherry blossoms in Tokyo and books their trip for the last week of April. What is the most likely outcome?",
        ["The blossoms will have already fallen — Tokyo's peak bloom is typically late March to early April", "Perfect timing — late April is consistently peak bloom in Tokyo", "The blossoms won't have opened yet — Tokyo's bloom starts in May", "Late April is ideal because the government schedules bloom for Golden Week"],
        0,
    ),

    # ── Gotcha questions: New Zealand ────────────────────────────────────────
    (
        "On New Zealand's Milford Track during the Great Walks season, independent hikers must:",
        ["Complete the track in exactly 4 days in one direction only, sleeping at assigned huts each night", "Choose either direction and book huts at least 24 hours in advance", "Join a guided group — independent hiking is not permitted on the Milford Track", "Complete the track within 7 days with flexible hut choices"],
        0,
    ),

    # ── Gotcha questions: Iceland ────────────────────────────────────────────
    (
        "Before entering any Icelandic public swimming pool or geothermal bath (including the Blue Lagoon), visitors are required to:",
        ["Shower naked without a swimsuit, washing specific body parts as shown on posted diagrams", "Rinse off briefly in their swimsuit under a shower head", "Apply a provided mineral balm to protect skin from geothermal water", "Pass a health screening at the reception desk"],
        0,
    ),

    # ── Gotcha questions: Africa & Middle East ───────────────────────────────
    (
        "What surprises most first-time visitors about the location of the Pyramids of Giza?",
        ["The urban sprawl of Cairo extends right to the edge of the pyramid complex", "The pyramids are a 3-hour drive into the Sahara from Cairo", "The pyramids are surrounded by a dense forest preserve", "The pyramids can only be reached by crossing the Nile by boat"],
        0,
    ),
    (
        "The Great Wildebeest Migration in East Africa is best described as:",
        ["A continuous year-round circular movement across the Serengeti-Mara ecosystem", "A single annual crossing of the Mara River in September", "A migration from South Africa to Kenya that happens every five years", "A three-month journey from the Serengeti to the Okavango Delta"],
        0,
    ),
    (
        "At the ancient city of Petra in Jordan, the famous Treasury (Al-Khazneh) facade is:",
        ["The first major structure visitors encounter, with the vast majority of the site beyond it", "The largest and most impressive structure in the entire site", "Located at the far end of the site, requiring a full day of hiking to reach", "The only carved facade that remains intact"],
        0,
    ),
    (
        "A traveler flying from Brazil to Tanzania via Addis Ababa may be denied entry to Tanzania if they lack which document?",
        ["A yellow fever vaccination certificate, required when transiting from endemic countries", "A negative COVID-19 PCR test taken within 24 hours", "A pre-approved Tanzania e-visa with a QR code", "A certified letter from their employer stating the purpose of travel"],
        0,
    ),
]


class TravelKnowledge(Task):

    letters = ('A', 'B', 'C', 'D')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Shuffle choices deterministically so correct answer is evenly
        # distributed across A/B/C/D (avoids always-pick-A bias).
        rng = random.Random(42)
        self.questions = []
        for question, choices, correct_idx in TRAVEL_QUESTIONS:
            correct_answer = choices[correct_idx]
            shuffled = choices[:]
            rng.shuffle(shuffled)
            new_idx = shuffled.index(correct_answer)
            self.questions.append((question, shuffled, new_idx))

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.questions)

    def get_example(self, index):
        question, choices, correct_idx = self.questions[index]
        assert len(choices) == 4, f"Question {index} must have exactly 4 choices"
        assert 0 <= correct_idx < 4, f"Question {index} has invalid correct_idx {correct_idx}"
        user_message = render_mc(question, self.letters, choices)
        assistant_message = self.letters[correct_idx]
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
        return {
            "messages": messages,
            "letters": self.letters,
        }

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in self.letters, \
            f"Travel answer {assistant_response} is expected to be one of {self.letters}"
        expected = conversation['messages'][-1]['content']
        return assistant_response == expected
