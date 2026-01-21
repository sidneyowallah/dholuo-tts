import pandas as pd
import string
import os
import torch
from transformers import pipeline
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
from transformers import AutoTokenizer, AutoModelForTokenClassification

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

def get_pos_enhanced_text(text):
    """
    Cleans punctuation and adds POS tags to every word.
    Format: Word|TAG
    """
    if not text or pd.isna(text):
        return ""
    
    try:
        # 1. Clean the raw text to match word-for-word
        words = text.split()
        tagged_sentence = []
        
        # 2. Run the pipeline with NO aggregation 
        # This allows us to manually align subwords back into whole words
        results = tagger(text)
        
        # Alignment Logic: We want to match model output back to the original words
        # A simple way for TTS is to tag the full word based on its first subword
        for word in words:
            clean_word = word.strip().lower().translate(str.maketrans('', '', string.punctuation))
            if not clean_word:
                continue
                
            # Find the tag for this specific word
            # We look for the first result that matches the start of our word
            tag = "UNK"
            for res in results:
                if res['word'].strip().replace(" ", "").lower() in clean_word:
                    tag = res['entity_group']
                    break
            
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