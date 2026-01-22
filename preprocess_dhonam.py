import pandas as pd
import string
import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION & HARDWARE
# ==========================================
INPUT_CSV = "data/csv/final_dataset.csv"             # Path to your DhoNam CSV
MODEL_PATH = "models/luo-pos"       # Path to your trained POS model
OUTPUT_METADATA = "data/csv/tts-metadata.csv"     # Final file for TTS training
CONFIDENCE_THRESHOLD = 80            # Optional: Filter for high-quality audio

# Detect Hardware
if torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple Silicon (MPS) for tagging...")
elif torch.cuda.is_available():
    device = 0
    print("Using Nvidia GPU (CUDA) for tagging...")
else:
    device = -1
    print("Using CPU for tagging...")

# ==========================================
# 2. INITIALIZE TAGGER (With Warning Fix)
# ==========================================

print("Loading Dholuo POS model and fixing tokenizer regex...")

# 1. Load the tokenizer explicitly with the suggested fix
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, 
    fix_mistral_regex=True
)

# 2. Load the model explicitly
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

# 3. Pass both to the pipeline
tagger = pipeline(
    "token-classification", 
    model=model, 
    tokenizer=tokenizer,      # Use the fixed tokenizer here
    aggregation_strategy="simple", 
    device=device
)
# ==========================================
# --- Key to Labels (POS Tags):-------
# NN: Noun (Nying)
# V: Verb (Tim)
# PRON: Pronoun (Maolo nying)
# ADJ: Adjective (Lero nying)
# ADV: Adverb (Lero tim)
# DET: Determiner (Kwalo)
# ADP: Adposition/Preposition (Chengo)
# CONJ: Conjunction (Riwo wach)
# PART: Particle (Ngero/Wach matin)
# ==========================================

DHOLUO_STOPWORDS = {
    # --- PRONOUNS (PRON) ---
    "an": "PRON", "emane": "PRON", "en": "PRON", "endo": "PRON", "gin": "PRON", 
    "in": "PRON", "ingi": "PRON", "mekgi": "PRON", "ne": "PRON", "nga": "PRON", 
    "ngama": "PRON", "ngata": "PRON", "un": "PRON", "wan": "PRON",

    # --- NOUNS (NN) ---
    "arus": "NN", "barua": "NN", "betri": "NN", "bote": "NN", "chamcham": "NN", 
    "chieng": "NN", "chieng'wa": "NN", "dala": "NN", "dend": "NN", "dhano": "NN", 
    "doo": "NN", "duond": "NN", "duong'": "NN", "dwe": "NN", "gima": "NN", 
    "gokinyi": "NN", "gweng": "NN", "higa": "NN", "homwake": "NN", "jangad": "NN", 
    "ji": "NN", "kauntiy": "NN", "kongo": "NN", "kuguok": "NN", "lee": "NN", 
    "lesonde": "NN", "lo": "NN", "lowo": "NN", "luwor": "NN", "meru": "NN", 
    "mutasi": "NN", "nam": "NN", "nairobi": "NN", "ndalo": "NN", "ngad": "NN", 
    "ngano": "NN", "ngat": "NN", "ng'at": "NN", "ngato": "NN", "ng'ato": "NN", 
    "ng'atno": "NN", "nge": "NN", "ngere": "NN", "nguon": "NN", "nguono": "NN", 
    "ngut": "NN", "ngwech": "NN", "nyak": "NN", "nyasaye": "NN", "nyathi": "NN", 
    "nying": "NN", "nyithindo": "NN", "nyoro": "NN", "okek": "NN", "ot": "NN", 
    "ouma": "NN", "piny": "NN", "protin": "NN", "rangi": "NN", "ranginy": "NN", 
    "richo": "NN", "sikul": "NN", "taa": "NN", "towel": "NN", "tudruok": "NN", 
    "wenyiew": "NN", "wiya": "NN", "yat": "NN", "chambwa": "NN",

    # --- VERBS (V) ---
    "bedoe": "V", "biro": "V", "chakre": "V", "cham": "V", "chamo": "V", 
    "chango": "V", "chano": "V", "chiemo": "V", "chiwo": "V", "chulo": "V", 
    "chuoyo": "V", "dhawo": "V", "dhiye": "V", "dino": "V", "dinuoya": "V", 
    "dongo": "V", "duoko": "V", "dwaro": "V", "gaye": "V", "gengo": "V", 
    "godoye": "V", "hero": "V", "hinya": "V", "kayawo": "V", "kendo": "V", 
    "kwayo": "V", "lam": "V", "lamo": "V", "limo": "V", "loso": "V", 
    "luok": "V", "luoro": "V", "luongo": "V", "luwo": "V", "miyo": "V", 
    "mobedo": "V", "modong'": "V", "mogik": "V", "moloyo": "V", "mondiki": "V", 
    "motegni": "V", "nang'iew": "V", "ndiko": "V", "neno": "V", "ngado": "V", 
    "nganyo": "V", "nganjo": "V", "ngieyo": "V", "ngiewo": "V", "nginjo": "V", 
    "ngiyo": "V", "ngol": "V", "ngolo": "V", "ngony": "V", "ngeyo": "V", 
    "ningogi": "V", "nitie": "V", "noduokie": "V", "nondiko": "V", "nwa": "V", 
    "nyalo": "V", "nyiewo": "V", "nyisi": "V", "parore": "V", "pidho": "V", 
    "pog": "V", "pongo": "V", "riembe": "V", "ringo": "V", "riwo": "V", 
    "romo": "V", "romre": "V", "tayowa": "V", "tedo": "V", "tim": "V", 
    "timo": "V", "timre": "V", "tingo": "V", "tiyogo": "V", "tony": "V", 
    "tugo": "V", "wuok": "V", "wuoth": "V", "yie": "V", "yudo": "V", 
    "yudore": "V",

    # --- ADJECTIVES (ADJ) ---
    "mongere": "ADJ", "ngeny": "ADJ", "ngich": "ADJ",

    # --- ADVERBS (ADV) ---
    "bange": "ADV", "bangeyo": "ADV", "chon": "ADV", "ka": "ADV", 
    "kane": "ADV", "kanyiyo": "ADV", "kapok": "ADV", "kariembo": "ADV", 
    "kinde": "ADV", "machiegni": "ADV", "machon": "ADV", "malo": "ADV", 
    "nene": "ADV", "ngar": "ADV", "nyime": "ADV",

    # --- DETERMINERS (DET) ---
    "duto": "DET", "ma": "DET", "mane": "DET", "moko": "DET", "yodno": "DET",

    # --- ADPOSITIONS (ADP) ---
    "e": "ADP", "gi": "ADP", "kaw": "ADP", "kod": "ADP", "kuom": "ADP", 
    "nyaka": "ADP",

    # --- CONJUNCTIONS (CONJ) ---
    "koso": "CONJ", "to": "CONJ",

    # --- PARTICLES (PART) ---
    "be": "PART", "bende": "PART", "ni": "PART", "ok": "PART"
}

def get_pos_enhanced_text(text):
    """
    Cleans punctuation and adds POS tags to every word.
    Format: Word|TAG
    """
    if not text or pd.isna(text):
        return ""
    
    try:
        # 1. Clean the raw text to match word-for-word
        raw_words = text.replace("’", "'").replace("‘", "'").split()
        tagged_sentence = []

        # Verb prefixes common the UNK list
        verb_prefixes = ('a', 'i', 'o', 'wa', 'u', 'gi', 'ko')
        
        # 2. Run the pipeline on the full text
        results = tagger(text)
        
        # Alignment Logic: We want to match model output back to the original words
        # A simple way for TTS is to tag the full word based on its first subword
        for word in raw_words:

            # Clean word for dictionary matching (remove punct, but keep internal ')
            clean_word = word.strip().lower().translate(str.maketrans('', '', string.punctuation.replace("'", "")))

            # Match word (remove ' for model token alignment i.e ng' to ng)
            match_word = clean_word.replace("'", "")

            if not match_word: continue
                
            tag = "UNK"
            for res in results:
                model_token = res['word'].replace(" ", "").replace("'", "").lower()
                if model_token in match_word or match_word in model_token:
                    tag = res['entity_group']
                    break
            
            # --- FALLBACK LOGIC ---
            if tag == "UNK":
                # 1. Dictionary Check
                if clean_word in DHOLUO_STOPWORDS or match_word in DHOLUO_STOPWORDS:
                    tag = DHOLUO_STOPWORDS.get(clean_word, DHOLUO_STOPWORDS.get(match_word))
                
                # 2. "-RUOK" Suffix (Always Nouns) - e.g., tudruok, dongruok, dusruok
                elif clean_word.endswith("ruok"):
                    tag = "NN"

                # 3. "KO-" Prefix (Often Infinitive Verbs) - e.g., kowuok, kongo (verb sense)
                elif clean_word.startswith("ko") and len(clean_word) > 4:
                    tag = "V"
                
                # 4. "JO-" Prefix (People Nouns)
                elif clean_word.startswith("jo"):
                    tag = "NN"
                
                # 5. "MA-" Prefix (Adjectives)
                elif clean_word.startswith("ma") and len(clean_word) > 3:
                    tag = "ADJ"

                # 6. Subject Verb Conjugation (a-, wa-, etc.)
                elif clean_word.startswith(verb_prefixes) and (clean_word.endswith("o") or len(clean_word) > 4):
                    tag = "V"

                # 7. Catch 'ne', 'en', 'gi', 'ng'a'
                elif len(clean_word) <= 2:
                    tag = "PRON" # Catch 'ne', 'en', 'gi', 'ng'a'
                    
                # # 8. Final safety fallback
                # else:
                #     tag = "NN"  
            
            tagged_sentence.append(f"{clean_word}_{tag}")
        
        return " ".join(tagged_sentence)
    
    except Exception as e:
        # If a specific sentence fails, return empty to be filtered later
        return ""

# ==========================================
# 3. LOAD & FILTER DATASET
# ==========================================
print(f"Reading {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)

# Apply Confidence Filter
initial_count = len(df)
df = df[df['confidence'] >= CONFIDENCE_THRESHOLD]
filtered_count = len(df)

print(f"Filtered out {initial_count - filtered_count} rows with confidence < {CONFIDENCE_THRESHOLD}.")
print(f"Remaining rows to process: {filtered_count}")

# ==========================================
# 4. DATA TRANSFORMATION (LJSpeech Format)
# ==========================================
# Enable progress bar for the tagging process
tqdm.pandas()

print("Generating POS tags (this may take a few minutes)...")
# Create the 3 columns required for TTS
# 1. audio_id (filename without .wav)
df['audio_id'] = df['audio_name'].str.replace('.wav', '', regex=False)

# 2. raw_text (The original transcription with punctuation)
df['raw_text'] = df['transcription']

# 3. pos_text (The normalized text with POS tags)
df['pos_text'] = df['transcription'].progress_apply(get_pos_enhanced_text)

# Final cleanup: Remove any rows where POS tagging failed
df = df[df['pos_text'] != ""]

# ==========================================
# 5. EXPORT
# ==========================================
# LJSpeech format: ID | Raw Text | Tagged Text
# We use the '|' separator which is standard for TTS
final_metadata = df[['audio_id', 'raw_text', 'pos_text']]

# Save to CSV
final_metadata.to_csv(
    OUTPUT_METADATA, 
    sep="|", 
    index=False, 
    header=False, 
    quoting=3,            # ensure quoting is turned OFF
    escapechar="\\"       # If a pipe ever appears in the text, it will be written as \|
)

print(f"\nSUCCESS!")
print(f"Final dataset contains {len(final_metadata)} pairs.")
print(f"File saved as: {OUTPUT_METADATA}")
print("-" * 30)
print("Next step: Move 'metadata.csv' and your '.wav' files into your TTS training folder.")