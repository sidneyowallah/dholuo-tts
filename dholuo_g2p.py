import re
import json
import pandas as pd
from tqdm import tqdm

class DholuoG2P:
    def __init__(self):
        # Order matters! Complex rules first (multi-char), then single chars.
        self.rules = [
            # 1. Digraphs & Special Consonants
            (r"ng'", "ŋ"),   # Velar nasal (ng'a)
            (r"ng", "ŋg"),   # Pre-nasalized (panga)
            (r"ny", "ɲ"),    # Palatal nasal (nyako)
            (r"th", "ð"),    # Dental fricative (thum) - varies by dialect, sometimes θ
            (r"dh", "ð"),    # Often voiced dental fricative
            (r"ch", "tʃ"),   # Affricate (chiro)
            (r"sh", "ʃ"),    # Fricative (shati)
            
            # 2. Vowels (Simple approx - Dholuo has ATR distinction, but standard IPA 'e/o' works for TTS start)
            (r"a", "a"),
            (r"e", "ɛ"),     # Open 'e' is common
            (r"i", "i"),
            (r"o", "ɔ"),     # Open 'o' is common
            (r"u", "u"),
            
            # 3. Standard Consonants
            (r"b", "b"), (r"p", "p"), (r"m", "m"), (r"w", "w"),
            (r"f", "f"), (r"v", "v"), (r"t", "t"), (r"d", "d"),
            (r"s", "s"), (r"n", "n"), (r"l", "l"), (r"r", "r"),
            (r"y", "j"), (r"k", "k"), (r"g", "g"), (r"h", "h"),
            (r"j", "dʒ")
        ]

    def predict(self, word):
        word = word.lower().strip()
        ipa = word
        
        for grapheme, phoneme in self.rules:
            ipa = re.sub(grapheme, phoneme, ipa)
            
        return ipa

# --- BULK GENERATOR ---
def generate_db_from_csv(csv_path, output_json="dholuo_pronunciation.json"):
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Extract all unique words from transcriptions
    all_text = " ".join(df['transcription'].dropna().astype(str).tolist())
    # Remove punctuation for word list
    import string
    translator = str.maketrans('', '', string.punctuation)
    clean_text = all_text.translate(translator).lower()
    
    unique_words = sorted(list(set(clean_text.split())))
    print(f"Found {len(unique_words)} unique words.")
    
    # 2. Convert all to IPA
    g2p = DholuoG2P()
    db = {}
    
    print("Generating IPA...")
    for word in tqdm(unique_words):
        # Logic: By default, assign the rule-based IPA to the "DEFAULT" key
        base_ipa = g2p.predict(word)
        
        # Structure it for your POS tagger
        db[word] = {
            "DEFAULT": base_ipa,
            # We leave specific V/NN tags empty for now unless we manually fix them later
            "V": base_ipa, 
            "NN": base_ipa 
        }
        
    # 3. Save
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=4, ensure_ascii=False)
    
    print(f"Saved {len(db)} entries to {output_json}")

if __name__ == "__main__":
    # Point this to your existing CSV
    generate_db_from_csv("data/csv/final_dataset.csv")