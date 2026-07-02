# =============================================================================
# knowledge/samphor_kb.py
# =============================================================================
# This file is the "answer book" — it stores all the actual reply text that
# the chatbot sends to the user.
#
# IMPORTANT: This file has NOTHING to do with machine learning.
# It is just a Python dictionary (a lookup table) that maps intent names to
# lists of human-written reply sentences.
#
# WHY A LIST OF REPLIES (not just one)?
#   Having 2–3 options per intent means the bot doesn't always say the exact
#   same sentence. MLEngine picks one at random using random.choice().
#
# HOW IT CONNECTS TO THE REST OF THE PROJECT:
#   - ml_engine.py imports INTENT_RESPONSES and FALLBACK_RESPONSE
#   - After the neural network decides WHICH intent the user asked about,
#     ml_engine.py looks up that intent name here and picks a reply.
#   - chat_engine.py (RuleBasedEngine) does NOT use this file — it has its
#     own inline reply strings inside its RULES dictionary.
#
# TO ADD A NEW TOPIC:
#   1. Add a new intent in data/intents.json  (with example questions)
#   2. Add a matching entry here              (with the reply text)
#   3. Re-run train.py to teach the model the new topic
# =============================================================================


# INTENT_RESPONSES is a dictionary.
# Key   = the intent tag string (must EXACTLY match the "tag" in intents.json)
# Value = a list of reply strings (at least 1 required; 2–3 is ideal)
INTENT_RESPONSES = {

    # ---------- What is the Samphor? ----------------------------------------
    "ask_definition": [
        "The Samphor (ស័មភោ) is a barrel-shaped, double-headed drum that serves as the rhythmic "
        "backbone of Cambodian classical music. It is roughly 50–60 cm long with two cow-skin heads "
        "of different diameters, producing two distinct pitches.",
        "A Samphor is a traditional Khmer drum with a wooden barrel body and two skin-covered heads. "
        "It is struck with both hands and guides the tempo of the Pinpeat ensemble.",
    ],

    # ---------- History / origins -------------------------------------------
    "ask_history": [
        "The Samphor traces its origins to the Angkor Empire (802–1431 CE). Bas-reliefs at Angkor Wat "
        "and the Bayon temple depict musicians playing barrel drums very similar to the modern Samphor, "
        "confirming its role in royal court music for over a millennium.",
        "Historical evidence places the Samphor in Cambodian culture at least as far back as the 9th century. "
        "Sanskrit inscriptions and Khmer temple carvings both reference drum-led ensembles in royal ceremonies.",
    ],

    # ---------- What it is made of ------------------------------------------
    "ask_material": [
        "The shell of the Samphor is carved from a single piece of jackfruit (Artocarpus heterophyllus) wood, "
        "prized for its resonance and durability. The two heads are made from cow or goat skin, "
        "laced together with rattan or leather cords that run the full length of the barrel.",
        "Traditional Samphor craftsmanship uses jackfruit wood for the body. The skins are treated, "
        "stretched, and tensioned with interlaced cords. Black paste (a mixture of rice and ash) "
        "is sometimes applied to the smaller head to fine-tune pitch.",
    ],

    # ---------- Physical appearance / dimensions ----------------------------
    "ask_shape": [
        "The Samphor has a barrel (or bobbin) shape — wider in the middle and narrowing toward each end. "
        "Its two circular heads differ in size: the larger head (approx. 22 cm diameter) produces the bass tone, "
        "while the smaller head (approx. 18 cm) produces a higher pitch.",
        "In profile the Samphor resembles an hourglass with a wide waist. It sits horizontally on a low stand "
        "in front of the seated player, with the larger head facing right.",
    ],

    # ---------- How to play it ----------------------------------------------
    "ask_playing": [
        "The Samphor is played with bare hands. The right hand strikes the larger head with the full palm "
        "and individual fingers to produce bass tones and complex rolls. The left hand taps the smaller head "
        "with fingertips. Tension on the lacing can be adjusted mid-performance to alter pitch.",
        "Playing the Samphor requires coordinating both hands independently. The dominant hand drives the main "
        "beat while the other provides syncopation. A player's palm can dampen the skin to create muted tones, "
        "adding expressive dynamics.",
    ],

    # ---------- Tuning ------------------------------------------------------
    "ask_tuning": [
        "Tuning is achieved by tightening or loosening the rattan lacing that runs between the two heads. "
        "A black paste (kroeung) made from cooked rice and charcoal ash is applied to the center of one head "
        "to add weight and lower its pitch, allowing precise tonal adjustment.",
        "The Samphor has no fixed Western pitch; its tuning is relative to the ensemble. "
        "Performers adjust the lacing tension before playing and may retune between pieces "
        "if the ambient temperature and humidity cause the skin to shift.",
    ],

    # ---------- Ceremonial use ----------------------------------------------
    "ask_ceremonies": [
        "The Samphor is indispensable in Cambodian religious and royal ceremonies: it accompanies "
        "Buddhist temple rituals, Khmer New Year celebrations, royal ploughing ceremonies, "
        "traditional weddings, and cremation rites.",
        "In ceremonial contexts the Samphor does more than keep time — specific rhythmic patterns "
        "(called 'choan') signal transitions between ritual phases, alerting participants and "
        "officiants to proceed to the next stage.",
    ],

    # ---------- The Pinpeat ensemble ----------------------------------------
    "ask_pinpeat": [
        "The Pinpeat is Cambodia's classical court orchestra, typically consisting of: "
        "Samphor (barrel drum), Skor Thom (large drums), Roneat Ek & Thung (xylophones), "
        "Kong Vong (gong circles), Sralai (quadruple-reed oboe), and Chhing (cymbals). "
        "The Samphor acts as the conductor, setting tempo and signalling form changes.",
        "Within the Pinpeat ensemble, the Samphor player holds a leadership role similar to a conductor. "
        "The patterns they play dictate the overall rhythmic cycle (choun) that all other instruments follow.",
    ],

    # ---------- Comparison with other drums ---------------------------------
    "ask_compare": [
        "Compared to the Indian tabla, the Samphor is a single-piece barrel drum rather than two separate drums; "
        "it is louder and less pitch-flexible but provides powerful rhythmic drive. "
        "Compared to the Japanese taiko, the Samphor is much smaller and played with hands rather than sticks.",
        "The Samphor shares its barrel shape with the Korean buk and the Sri Lankan yak bera, "
        "reflecting ancient trade and cultural exchange across Southeast and South Asia. "
        "However, its interlaced-cord tensioning system is distinctively Khmer.",
    ],

    # ---------- How to learn ------------------------------------------------
    "ask_learning": [
        "Learning the Samphor traditionally begins around age 8-10 at pagoda schools or arts conservatories "
        "such as the Royal University of Fine Arts in Phnom Penh. Students first master basic strokes, "
        "then progress to fixed rhythmic cycles before learning the full repertoire.",
        "The Royal University of Fine Arts (RUFA) in Phnom Penh offers formal Samphor training. "
        "Several NGOs, including Cambodian Living Arts, also run community programs "
        "to ensure knowledge transfer to younger generations.",
    ],

    # ---------- Preservation / cultural survival ----------------------------
    "ask_preservation": [
        "After the Khmer Rouge genocide (1975–1979) decimated the artistic community, "
        "surviving masters worked with UNESCO and the Royal Government of Cambodia to revive "
        "Pinpeat music. The Samphor tradition is now listed on UNESCO's Intangible Cultural Heritage list.",
        "Preservation efforts include: master-apprentice programs at RUFA, digital archiving of "
        "performances, instrument-making workshops in rural communities, and diaspora teaching programs "
        "in the United States, France, and Australia.",
    ],

    # ---------- Greeting ----------------------------------------------------
    "greeting": [
        "Hello! I'm the Samphor Expert. Ask me anything about this beautiful Cambodian drum — "
        "its history, how it's made, how it's played, or its cultural significance.",
        "Greetings! Welcome to the Samphor Expert System. How can I help you learn about Cambodia's iconic drum today?",
        "Hi there! Ready to explore the world of the Samphor? Ask me anything!",
    ],

    # ---------- Farewell ----------------------------------------------------
    "farewell": [
        "Goodbye! May the rhythm of the Samphor stay with you.",
        "Farewell! Thank you for learning about Cambodian musical heritage.",
        "See you next time! Keep the tradition alive.",
    ],

    # ---------- Out-of-scope (topics the bot doesn't know about) -----------
    "out_of_scope": [
        "I'm specialised in the Samphor drum and Cambodian classical music. "
        "I don't have information on that topic. Try asking about the drum's history, materials, playing technique, "
        "tuning, or its role in ceremonies.",
        "That's outside my area of expertise. I can answer questions about the Samphor — "
        "what it is, how it's played, its cultural role, or preservation efforts.",
    ],
}


# FALLBACK_RESPONSE is used by ml_engine.py when the neural network's confidence
# score is below 0.7 — meaning the model isn't sure which category fits best.
FALLBACK_RESPONSE = (
    "I'm not confident I understood that. Could you rephrase? "
    "You can ask about the Samphor's definition, history, materials, shape, playing technique, "
    "tuning, ceremonies, the Pinpeat ensemble, comparisons, learning, or preservation."
)

# INTENT_LABELS maps internal tag names to plain-English topic descriptions.
# Used by ml_engine.py to build human-readable clarification questions when
# the model is uncertain between two candidate intents.
# Fix 13 — Khmer-language responses served when the user writes in Khmer script.
# Keys match intent tags in INTENT_RESPONSES above.
KHMER_RESPONSES: dict[str, list[str]] = {

    "ask_definition": [
        "ស័មភោ (ស័មភោ) គឺ ជា ស្ករ ពីរ ក្បាល ដែល ជា ស្នូល ចង្វាក់ ក្នុង ភ្លេង ខ្មែរ បុរាណ។ "
        "ស្ករ នេះ ធ្វើ ឡើង ពី ឈើ ខ្ទមសែ ហើយ មាន ស្បែក គោ ឬ ពពែ នៅ ចុង ទាំង ពីរ។ "
        "វា មាន ប្រវែង ប្រហែល ៥០–៦០ ស.ម។",
        "ស័មភោ ជា ឧបករណ៍ ដំ ប្រពៃណី ខ្មែរ ដែល ប្រើ ក្នុង ភ្លេង ចាក្រព ខ្មែរ "
        "និង ក្នុង ពិណ ពាទ្យ។ ស្ករ នេះ មាន ក្បាល ពីរ ដែល ខ្នាត ខ្លះ ខ្នាត ទ្រ ខ្លះ។",
    ],

    "ask_history": [
        "ស័មភោ មាន ប្រវត្តិ ចាប់ ពី សម័យ អង្គរ (ឆ្នាំ ៨០២–១៤៣១)។ "
        "រូប ចម្លាក់ ស្ករ នេះ ត្រូវ បាន ឃើញ នៅ ប្រាសាទ អង្គរ វត្ត និង ប្រាសាទ បាយ័ន "
        "ដែល បញ្ជាក់ ថា ស្ករ ខ្មែរ មាន អាយុ ជាង ១ ០០០ ឆ្នាំ ហើយ។",
        "ការ ស្រាវជ្រាវ ប្រវត្តិ សាស្ត្រ បញ្ជាក់ ថា ស័មភោ ត្រូវ បាន ប្រើ ក្នុង ព្រះ ពូជ ខ្មែរ "
        "ចាប់ ពី សតវត្ស ទី ៩ ម.គ.ស.។ ស្ករ នេះ ដើរ តួ ជា ឧបករណ៍ ភ្លេង ក្នុង ព្រះ រាជ ព្រំ ដំ ចាំ ថ្ងៃ។",
    ],

    "ask_material": [
        "ស័មភោ ធ្វើ ឡើង ពី ឈើ ខ្ទមសែ (Artocarpus heterophyllus) ដែល ល្បី ព្រោះ ជ្រៅ ថ្ងន់ ល្អ។ "
        "ក្បាល ស្ករ ធ្វើ ពី ស្បែក គោ ឬ ស្បែក ពពែ ហើយ ចង ជាប់ ដោយ ចំណង រណ្ដំ ឬ ស្បែក តែ ម្ដង។",
        "ច្នៃ ប្រឌិត ស័មភោ ប្រើ ឈើ ខ្ទមសែ ដែល ចោះ ពី ឈើ ដ ព ម្ដង ដើម្បី បាន ប្រអប់ ស្ករ មួយ។ "
        "ស្បែក គោ ត្រូវ បាន ស្ករ ហើយ ផ្ចង់ ទៅ លើ ទទឹង ចុង ពីរ ដោយ ប្រើ ចំណង រណ្ដំ។",
    ],

    "ask_playing": [
        "ស័មភោ លេង ដោយ ប្រើ ដៃ ទទេ ប្រ ការ ដំ។ ដៃ ស្ដាំ ដំ ក្បាល ធំ ឲ្យ ឮ សំឡេង ទន់ ថ្មើរ "
        "ហើយ ដៃ ឆ្វេង ម្រាម ដៃ ដំ ក្បាល តូច ឲ្យ ស្ដាប់ ឮ សំឡេង ខ្ពស់ ជាង ។",
        "ការ លេង ស័មភោ ត្រូវ ប្រើ ដៃ ទាំង ពីរ ដោយ ចម្លែក ពី គ្នា។ "
        "ស្ករ ខ្មែរ នេះ ដើរ ជា ចង្វាក់ ចាក្រព ក្នុង ពិណ ពាទ្យ ខ្មែរ។",
    ],

    "ask_ceremonies": [
        "ស័មភោ ត្រូវ ប្រើ ក្នុង ពិធី ជាច្រើន ដូច ជា ពិធី ចូល ឆ្នាំ ខ្មែរ ពិធី អាពាហ៍ពិពាហ៍ "
        "ពិធី ខ្មោច ខ្មែរ និង ពិធីកម្ម ព្រះ ពុទ្ធ សាសនា នៅ វត្ត ។",
        "ក្នុង ពិធីការ ស័មភោ ធ្វើ ជា អ្នក ដឹកនាំ ចង្វាក់ ហើយ ប្រើ ស្ទ្រង់ ចុះ ឡើង (ចំណង) "
        "ដែល ជំ នួន ដំណើរ ការ ពិធី ទៅ ដំណាក់ ថ្មី ។",
    ],

    "ask_pinpeat": [
        "ពិណ ពាទ្យ គឺ ជា វង់ ភ្លេង ចាក្រព ខ្មែរ ដែល មាន ស័មភោ រនាត រ៉ង ស្ករ ធំ ខ្នាក់ "
        "ស្រឡៃ និង ភ្លុក ។ ស័មភោ ជា អ្នក ដឹកនាំ ចង្វាក់ ភ្លេង ទាំង មូល ក្នុង វង់ ភ្លេង នេះ ។",
        "ស័មភោ ដើរ ជា អ្នក ចង្វាក់ ពិណ ពាទ្យ ។ ឧបករណ៍ ផ្សេង ៗ ដូច ជា រ នាត ឯក រ នាត ធុង "
        "រ៉ង វ៉ង ភ្លុក ស្រឡៃ ត្រូវ ធ្វើ តាម ចង្វាក់ ដែល ស័មភោ កំណត់ ។",
    ],

    "ask_learning": [
        "ជា ប្រពៃណី ការ រៀន ស័មភោ ចាប់ ផ្ដើម ពី អាយុ ៨–១០ ឆ្នាំ នៅ ក្នុង វត្ត "
        "ឬ នៅ មហាវិទ្យាល័យ វិចិត្រ សិល្ប៍ ភ្នំ ពេញ (RUFA) ។ "
        "ការ បង្ហាត់ ចូល ចិត្ត ប្រើ ប្រព័ន្ធ ឆ្ល ើយ ឆ្លង អ្នក គ្រូ-សិស្ស ។",
        "ដើម្បី រៀន ស័មភោ ត្រូវ ជ្ញ ា ប ការ ដំ ជា មុន ហើយ ក្រោយ មក ទៅ រៀន ទំ នុក ចង្វាក់ "
        "ពេញ ។ ខ្ញុំ ណែ នាំ ឱ្យ ទៅ ព្រះ ទ្រ RUFA ក្នុង ភ្នំ ពេញ ។",
    ],

    "ask_preservation": [
        "ក្រោយ ពី ការ ប្រហារ ប្រជាជន ខ្មែរ ក្រហម (១៩៧៥–១៩៧៩) ស ន្ត តំ ណ តន្ត្រី ខ្មែរ ត្រូវ "
        "ស ន្ដ ប ំ ឡើង វិញ ដោយ ជំ នួ យ ជា មួយ UNESCO ។ ស័មភោ ឥ ឡូ វ នេះ ជា ភ្នំ ជើង "
        "ក្នុង បញ្ជី ស្នា ដៃ វប្ប ធម៌ អ រូបី UNESCO ។",
        "អង្គ ការ Cambodian Living Arts និង RUFA ខ ច ំ ខ ន ការ អប់ រំ ស័មភោ ។ "
        "ការ ថ ត ភ្លេង ស ង្គ្រោះ នា ពេ ល ខ្លះ ត្រូ វ ប ន្ថែ ម ដើម្បី ការ ព ណ៌ ។",
    ],

    "greeting": [
        "សួស្ដី! ខ្ញុំ ជា អ្នក ជំ នាញ ស័មភោ ។ សូម សួរ ខ្ញុំ ពី ស្ករ ប្រពៃណី ខ្មែរ នេះ!",
        "ជំ រាប សួរ! ស ូ ម ស្វាគ មន៍ ។ ខ្ញុំ ត្រៀម ខ្លួន ជ ួ យ អ្ន ក ស្វែ ង យល់ ពី ស័ម ភោ ។",
    ],

    "farewell": [
        "លា ហើយ! ស ូ ម ឲ្យ ចង្វាក់ ស័ម ភោ ស្ថិ ត ជាមួ យ អ្ន ក ។",
        "ជំ រាប លា! អ រ គុ ណ ដែ ល ស ិ ក្សា ពី ប ប្ប ធ ម៌ ខ្មែ រ ។",
    ],
}


# Fix 7 — Rich media: Wikipedia / external links shown below bot replies.
# Key = intent tag; value = dict with optional "links" list of {label, url}.
RICH_MEDIA: dict[str, dict] = {
    "ask_definition": {"links": [
        {"label": "Samphor — Wikipedia", "url": "https://en.wikipedia.org/wiki/Samphor"},
        {"label": "Music of Cambodia — Wikipedia", "url": "https://en.wikipedia.org/wiki/Music_of_Cambodia"},
    ]},
    "ask_history": {"links": [
        {"label": "Khmer Empire — Wikipedia", "url": "https://en.wikipedia.org/wiki/Khmer_Empire"},
        {"label": "Angkor Wat — Wikipedia", "url": "https://en.wikipedia.org/wiki/Angkor_Wat"},
    ]},
    "ask_material": {"links": [
        {"label": "Jackfruit — Wikipedia", "url": "https://en.wikipedia.org/wiki/Jackfruit"},
    ]},
    "ask_playing": {"links": [
        {"label": "Pinpeat ensemble — Wikipedia", "url": "https://en.wikipedia.org/wiki/Pinpeat"},
    ]},
    "ask_tuning": {"links": [
        {"label": "Drum tuning — Wikipedia", "url": "https://en.wikipedia.org/wiki/Drum_tuning"},
    ]},
    "ask_ceremonies": {"links": [
        {"label": "Khmer New Year — Wikipedia", "url": "https://en.wikipedia.org/wiki/Khmer_New_Year"},
        {"label": "Buddhism in Cambodia — Wikipedia", "url": "https://en.wikipedia.org/wiki/Buddhism_in_Cambodia"},
    ]},
    "ask_pinpeat": {"links": [
        {"label": "Pinpeat — Wikipedia", "url": "https://en.wikipedia.org/wiki/Pinpeat"},
        {"label": "Cambodian classical dance — Wikipedia", "url": "https://en.wikipedia.org/wiki/Cambodian_classical_dance"},
    ]},
    "ask_compare": {"links": [
        {"label": "Tabla — Wikipedia", "url": "https://en.wikipedia.org/wiki/Tabla"},
        {"label": "Taiko — Wikipedia", "url": "https://en.wikipedia.org/wiki/Taiko"},
    ]},
    "ask_learning": {"links": [
        {"label": "Royal University of Fine Arts — Wikipedia", "url": "https://en.wikipedia.org/wiki/Royal_University_of_Fine_Arts"},
        {"label": "Cambodian Living Arts", "url": "https://cambodianlivingarts.org"},
    ]},
    "ask_preservation": {"links": [
        {"label": "UNESCO Intangible Cultural Heritage", "url": "https://ich.unesco.org/en/RL/royal-ballet-of-cambodia-00097"},
        {"label": "Khmer Rouge — Wikipedia", "url": "https://en.wikipedia.org/wiki/Khmer_Rouge"},
    ]},
}


INTENT_LABELS: dict[str, str] = {
    "ask_definition":   "what the Samphor is",
    "ask_history":      "the history of the Samphor",
    "ask_material":     "the materials it is made from",
    "ask_shape":        "its physical shape and dimensions",
    "ask_playing":      "how to play it",
    "ask_tuning":       "how it is tuned",
    "ask_ceremonies":   "its ceremonial and ritual uses",
    "ask_pinpeat":      "the Pinpeat ensemble",
    "ask_compare":      "how it compares to other drums",
    "ask_learning":     "how to learn the Samphor",
    "ask_preservation": "preservation and cultural survival efforts",
    "greeting":         "a greeting",
    "farewell":         "a farewell",
    "out_of_scope":     "an off-topic question",
}
