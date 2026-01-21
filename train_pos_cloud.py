# =========================================================
# This script provides a technical framework for fine-tuning a Transformer model specifically for Part-of-Speech (POS) tagging using the KenPOS dataset. 
# The process begins by restructuring raw data from a row-based format into grouped sentences and mapping text labels to numerical identifiers. 
# It utilizes a pre-trained Afro-XLMR model and includes a specialized function to align word-level tags with subword tokens generated during processing. 
# The configuration is optimized for high-performance hardware, employing mixed precision training and specific learning rates over multiple epochs. 
# Finally, the code executes the training routine through a standardized trainer interface and exports the refined model for future linguistic analysis.
# ==========================================================


import os
import pandas as pd
import torch
from datasets import Dataset, Features, Sequence, Value, ClassLabel
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification
)

# ==========================================
# 1. LOAD AND PREPARE LOCAL DATA
# ==========================================
print("Loading KenPOS dho.parquet...")
df = pd.read_parquet("dho.parquet")

# KenPOS is usually stored as one token per row. 
# We must group them by sentence_id to create full sentences.
grouped = df.groupby("sentence_id").agg({
    "token": list,
    "pos_tag": list
}).reset_index()

# Get unique labels and create mappings
unique_tags = sorted(list(set(df["pos_tag"].unique())))
label2id = {tag: i for i, tag in enumerate(unique_tags)}
id2label = {i: tag for tag, i in label2id.items()}

print(f"Unique POS tags found: {unique_tags}")

# Define the Schema (Features)
# This ensures the dataset is structured correctly for Transformers
features = Features({
    "token": Sequence(Value("string")),
    "tags": Sequence(ClassLabel(num_classes=len(unique_tags), names=unique_tags)),
    "sentence_id": Value("int64"),
    "pos_tag": Sequence(Value("string")) # Keep original strings for reference
})

def map_labels_to_ids(example):
    example["tags"] = [label2id[tag] for tag in example["pos_tag"]]
    return example

# Convert Pandas to Hugging Face Dataset
raw_dataset = Dataset.from_pandas(grouped)
raw_dataset = raw_dataset.map(map_labels_to_ids)
dataset = raw_dataset.cast(features)

# Split into 90% training, 10% validation
dataset = dataset.train_test_split(test_size=0.1)

# ==========================================
# 2. TOKENIZATION & SUBWORD ALIGNMENT
# ==========================================
model_id = "Davlan/afro-xlmr-large-76L"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize_and_align_labels(examples):
    # 'is_split_into_words=True' is critical for POS tagging
    tokenized_inputs = tokenizer(
        examples["token"], 
        truncation=True, 
        is_split_into_words=True, 
        max_length=512
    )

    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            # Special tokens (like <s>) get -100 so they are ignored by the loss function
            if word_idx is None:
                label_ids.append(-100)
            # Only label the FIRST subword of a word
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # Subsequent subwords (e.g., 'nyu', '##mba') get -100
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# ==========================================
# 3. MODEL INITIALIZATION
# ==========================================
model = AutoModelForTokenClassification.from_pretrained(
    model_id, 
    num_labels=len(unique_tags),
    id2label=id2label,
    label2id=label2id
)

# ==========================================
# 4. TRAINING CONFIGURATION (RunPod Optimized)
# ==========================================
training_args = TrainingArguments(
    output_dir="/workspace/luo-pos-tagger",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8, # Adjust based on GPU VRAM
    per_device_eval_batch_size=8,
    num_train_epochs=10,           # More epochs for small specific datasets
    weight_decay=0.01,
    fp16=True,                     # Uses Mixed Precision for speed on RTX/A100 GPUs
    logging_steps=50,
    save_total_limit=1,
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to="none"               # Prevents requiring WandB login
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    processing_class=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
)

# ==========================================
# 5. EXECUTE TRAINING
# ==========================================
print("Starting training...")
trainer.train()

# Save the final model
trainer.save_model("/workspace/luo-pos-final")
print("Model saved to /workspace/luo-pos-final")