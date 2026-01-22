import pandas as pd
import json

# Load your POS-aware dictionary
with open("data/dholuo_lexicon.json", "r", encoding="utf-8") as f:
    lexicon = json.load(f)

# Load your current metadata
df = pd.read_csv("data/csv/tts-metadata.csv", sep="|", header=None, names=["id", "raw", "tagged"])
def convert_to_ipa(tagged_text):
    tokens = tagged_text.split()
    ipa_sentence = []
    for token in tokens:
        # Look up word_TAG in our lexicon
        # If not found, we use the word part as a fallback
        ipa = lexicon.get(token, token.split("_")[0])
        ipa_sentence.append(ipa)
    return " ".join(ipa_sentence)

print("Converting tagged text to POS-aware IPA...")
df["phonemes"] = df["tagged"].apply(convert_to_ipa)

# Save as final training metadata
# Format: ID | Raw_Text | Phoneme_Text
df[["id", "raw", "phonemes"]].to_csv("data/csv/train_metadata.csv", sep="|", index=False, header=False)
print("Created train_metadata.csv")