import pandas as pd
import json

# Load your POS-aware dictionary
with open("data/dholuo_lexicon.json", "r", encoding="utf-8") as f:
    lexicon = json.load(f)

# Load gender mapping
users = pd.read_csv('data/csv/users-meta.csv')
if 'user-id' in users.columns:
    users = users.rename(columns={'user-id': 'user_id'})
gender_map = dict(zip(users['user_id'], users['gender']))

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

# Add gender column
df['user_id'] = df['id'].str.rsplit('-', n=1).str[0]
df['gender'] = df['user_id'].map(gender_map)

# Save combined metadata
df[["id", "raw", "phonemes"]].to_csv("data/csv/train_metadata.csv", sep="|", index=False, header=False)
print(f"Created train_metadata.csv ({len(df)} samples)")

# Split by gender
df_male = df[df['gender'] == 'male_masculine'][["id", "raw", "phonemes"]]
df_female = df[df['gender'] == 'female_feminine'][["id", "raw", "phonemes"]]

df_male.to_csv("data/csv/male_training_metadata.csv", sep="|", index=False, header=False)
df_female.to_csv("data/csv/female_training_metadata.csv", sep="|", index=False, header=False)

print(f"Created male_training_metadata.csv ({len(df_male)} samples)")
print(f"Created female_training_metadata.csv ({len(df_female)} samples)")