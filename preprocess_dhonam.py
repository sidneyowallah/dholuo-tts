import pandas as pd
import os
from tqdm import tqdm
from tagger import Tagger

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

def get_pos_enhanced_text(text, tagger):
    """
    Uses the Tagger class to tag text.
    Format: Word_TAG
    """
    if not text or pd.isna(text):
        return ""
    
    try:
        tagged_pairs = tagger.tag(text)
        return " ".join([f"{word}_{tag}" for word, tag in tagged_pairs])
    except Exception:
        return ""

def main():
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    INPUT_CSV = "data/csv/final_dataset.csv"
    MODEL_PATH = "models/luo-pos"
    OUTPUT_METADATA = "data/csv/tts-metadata.csv"
    CONFIDENCE_THRESHOLD = 80

    # ==========================================
    # 2. INITIALIZE TAGGER
    # ==========================================
    print("Loading Dholuo POS tagger...")
    tagger = Tagger(model_path=MODEL_PATH)

    # ==========================================
    # 3. LOAD & FILTER DATASET
    # ==========================================
    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    # Apply Confidence Filter
    if CONFIDENCE_THRESHOLD:
        df = df[df['confidence'] >= CONFIDENCE_THRESHOLD]
        print(f"Filtered to {len(df)} high-confidence samples (>={CONFIDENCE_THRESHOLD}%)")

    # ==========================================
    # 4. APPLY POS TAGGING
    # ==========================================
    print("Applying POS tags to transcripts...")
    tqdm.pandas(desc="Tagging")
    df['pos_tagged'] = df['transcript'].progress_apply(lambda x: get_pos_enhanced_text(x, tagger))

    # Remove failed rows
    df = df[df['pos_tagged'].str.len() > 0]
    print(f"Successfully tagged {len(df)} samples")

    # ==========================================
    # 5. SAVE METADATA
    # ==========================================
    output_df = df[['audio_file', 'transcript', 'pos_tagged']]
    output_df.to_csv(OUTPUT_METADATA, sep='|', header=False, index=False)
    print(f"\nSaved TTS metadata to {OUTPUT_METADATA}")
    print(f"Format: audio_file|transcript|pos_tagged")

if __name__ == "__main__":
    main()
