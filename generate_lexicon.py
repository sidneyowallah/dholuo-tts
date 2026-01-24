import json
import pandas as pd
from tqdm import tqdm
from phonemizer import Phonemizer

# --- BULK GENERATOR ---
def generate_g2p_from_metadata(metadata_path, output_json="data/dholuo_lexicon.json"):
    # 1. Load the metadata.csv (3 columns, pipe-separated)
    print(f"Reading {metadata_path}...")
    df = pd.read_csv(metadata_path, sep="|", header=None, names=["id", "raw", "pos"])
    
    g2p = Phonemizer(tagger=False)
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
            
            # Use the phonemizer to get POS-aware IPA
            ipa_final = g2p.phonemize(word, tag)
            
            # Store it exactly as it appears in metadata (e.g., "nam_NN")
            lexicon[token] = ipa_final

    # 4. Save the dictionary
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(lexicon, f, indent=4, ensure_ascii=False)
    
    print(f"Saved {len(lexicon)} POS-aware entries to {output_json}")

if __name__ == "__main__":
    # RUN IT
    generate_g2p_from_metadata("data/csv/tts-metadata.csv")