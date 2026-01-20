import pandas as pd

# Load dataset
df = pd.read_parquet('data/dho.parquet')

# Count POS tag frequencies
pos_counts = df['pos_tag'].value_counts()
print(pos_counts.head(10))
print(f"Total unique POS tags: {len(pos_counts)}")

print("\n===========================================\n")
# Get first sentence
sentence_0 = df[df['sentence_id'] == 0].sort_values('position')
print(' '.join(sentence_0['token'].tolist()))