import re
import json
import pandas as pd
from tqdm import tqdm

class DholuoG2P:
    def __init__(self):
        # Map complex digraphs first to avoid partial matching
        # Use special characters for phonemes to avoid "double-dipping"
        self.mapping = {
            # 1. Digraphs & Special Consonants
            "ng'": "ŋ",    # Velar nasal (ng'a)
            "ng": "ŋɡ",    # Pre-nasalized (panga)
            "ny": "ɲ",     # Palatal nasal (nyako)
            "th": "θ",     # Dental fricative (thum) - varies by dialect, sometimes θ
            "dh": "ð",     # dho (voiced dental)
            "ch": "tʃ",    # Affricate (chiro)
            "sh": "ʃ",     # Fricative (shati)
            "j": "ɟ",      # voiced palatal plosive (standard Dholuo 'j')
            "y": "j",      # semi-vowel 'y' -> IPA 'j'
            
            # 2. Vowels (Simple approx - Dholuo has ATR distinction, but standard IPA 'e/o' works for TTS start)
            "a": "a", 
            "e": "ɛ",      # Open 'e' is common
            "i": "i", 
            "o": "ɔ",      # Open 'o' is common
            "u": "u",

            # 3. Standard Consonants
            "b": "b", "p": "p", "m": "m", "w": "w",
            "f": "f", "v": "v", "t": "t", "d": "d",
            "s": "s", "n": "n", "l": "l", "r": "ɾ",
            "k": "k", "g": "ɡ", "h": "h"
        }
        # Compile a regex to match all keys, longest first
        pattern = "|".join(re.escape(k) for k in sorted(self.mapping.keys(), key=len, reverse=True))
        self.regex = re.compile(pattern)

    def predict(self, word):
        # This replaces everything in ONE pass, preventing double-processing
        return self.regex.sub(lambda m: self.mapping[m.group(0)], word.lower().strip())

# --- BULK GENERATOR ---
def generate_g2p_from_metadata(metadata_path, output_json="data/dholuo_lexicon.json"):
    # 1. Load the metadata.csv (3 columns, pipe-separated)
    print(f"Reading {metadata_path}...")
    df = pd.read_csv(metadata_path, sep="|", header=None, names=["id", "raw", "pos"])
    
    g2p = DholuoG2P()
    lexicon = {}
    
    print("Extracting Word_TAG pairs and generating IPA...")
    
    # 2. Iterate through the POS-tagged column (the 3rd column)
    for row in tqdm(df['pos'].dropna()):
        # Example row: "ne_UNK en_UNK jatend_NN nam_NN"
        tokens = row.split()
        
        for token in tokens:
            if "_" not in token:
                continue
                
            # Split into word and tag (e.g., 'nam' and 'NN')
            word, tag = token.rsplit("_", 1)
            
            # Use the G2P to get the base pronunciation
            ipa_base = g2p.predict(word)
            
            # 3. Create POS-aware IPA
            # We apply tones based on the tag
            if tag == "V":
                ipa_final = ipa_base + "˥" # High tone for Verbs
            elif tag == "NN":
                ipa_final = ipa_base + "˩" # Low tone for Nouns
            else:
                ipa_final = ipa_base         # Neutral for others
            
            # Store it exactly as it appears in metadata (e.g., "nam_NN")
            lexicon[token] = ipa_final

    # 4. Save the dictionary
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(lexicon, f, indent=4, ensure_ascii=False)
    
    print(f"Saved {len(lexicon)} POS-aware entries to {output_json}")

# RUN IT
generate_g2p_from_metadata("data/csv/tts-metadata.csv")