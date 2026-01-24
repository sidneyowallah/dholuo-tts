import pandas as pd
import os
from tqdm import tqdm
from tagger import Tagger, DHOLUO_STOPWORDS

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
