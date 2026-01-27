# demo/examples.py

EXAMPLES = [
    [
        "Nyithindo ringo e dala",
        "Common Phrase",
        "Children are running home (Tests: Noun-Verb agreement)"
    ],
    [
        "Oyawore! Iriyo nade?",
        "Greetings",
        "Good morning! How did you sleep? (Tests: Intonation)"
    ],
    [
        "Ouma ne ong'iewo gweno",
        "Past Tense",
        "Ouma bought a chicken (Tests: 'ne' auxiliary + verb)"
    ],
    [
        "An gi klamu",
        "Possession",
        "I have a pen (Tests: Pronoun + Preposition)"
    ],
    [
        "Dhok duto olal e thim",
        "Complex",
        "All the cows got lost in the forest (Tests: Plurals, Locatives)"
    ],
    [
        "In kanye?",
        "Question",
        "Where are you? (Tests: Low-High tone pattern)"
    ],
    [
        "Japuonj morwa mar bayoloji ne olero kaka ler loso gik moko ka itiyo gi picha ma oting'o weche mathoth",
        "Long Sentence",
        "Our biology teacher explained how light creates things using pictures containing many words (Tests: Breath groups)"
    ]
]

# Quick access dict for dropdowns if needed
EXAMPLE_MAP = {txt: desc for txt, cat, desc in EXAMPLES}
