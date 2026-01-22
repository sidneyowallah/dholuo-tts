import pandas as pd
from collections import Counter

# 1. Load the metadata
df = pd.read_csv("data/csv/tts-metadata.csv", sep="|", header=None, names=["id", "raw", "pos"])

all_unks = []

# 2. Extract every word tagged as _UNK
for row in df["pos"]:
    if pd.isna(row): continue
    
    tokens = row.split()
    for token in tokens:
        if "_UNK" in token:
            # Get word, lowercase it, and strip it
            word = token.split("_")[0].strip().lower()
            all_unks.append(word)

# 3. Create the unique counts
unk_counts = Counter(all_unks)
unique_words = sorted(list(unk_counts.keys()))

# 4. Results
print(f"--- UNK STATS ---")
print(f"Total _UNK occurrences: {len(all_unks)}")
print(f"Total UNIQUE words needing tags: {len(unique_words)}")
print(f"-----------------\n")

print("Top 10 most frequent unique UNKs (Priority for your dictionary):")
for word, count in unk_counts.most_common(10):
    print(f"  {word}: {count}")

print("\n--- COPY-PASTE LIST FOR YOUR SCRIPT ---")
print("Unique UNK List (Alphabetical):")
print(unique_words)