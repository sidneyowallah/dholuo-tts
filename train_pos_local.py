import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)
import evaluate

# 1. SETUP DEVICE & MEMORY SETTINGS
# Disable memory high-watermark limits to utilize full RAM if needed
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# 2. LOAD DATASET
data_path = "data/dho.parquet"

print(f"Loading dataset from: {data_path}...")
df = pd.read_parquet(data_path)

# --- RECONSTRUCT SENTENCES (Data Preprocessing) ---
# The dataset loads as "one word per row", so we must group them into sentences.
print("Grouping words into sentences...")

# Group by filename AND sentence_id to ensure sentences are separated correctly
df_grouped = df.groupby(['filename', 'sentence_id']).agg({
    'token': list,
    'pos_tag': list
}).reset_index()

# Rename columns to standard HuggingFace names
df_grouped.rename(columns={'token': 'tokens', 'pos_tag': 'ner_tags'}, inplace=True)

# Convert back to HuggingFace Dataset
full_dataset = Dataset.from_pandas(df_grouped)

# Create Train/Test split (80% train, 20% test)
dataset_split = full_dataset.train_test_split(test_size=0.2)
print(f"Data split: {len(dataset_split['train'])} training sentences, {len(dataset_split['test'])} test sentences.")

# 3. CREATE LABEL MAPPINGS
# Extract all unique tags (e.g., NOUN, VERB) to create ID maps
label_list = sorted(list(set(tag for row in dataset_split['train']['ner_tags'] for tag in row)))
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}
num_labels = len(label_list)

print(f"Found {num_labels} labels: {label_list}")

# 4. TOKENIZATION
# Switch to the MINI model for better performance/memory balance
model_checkpoint = "Davlan/afro-xlmr-base"
print(f"Loading model checkpoint: {model_checkpoint}")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens (CLS, SEP) get -100
            if word_idx is None:
                label_ids.append(-100)
            # Only label the first token of a word (subword alignment)
            elif word_idx != previous_word_idx:
                if label[word_idx] in label2id:
                    label_ids.append(label2id[label[word_idx]])
                else:
                    # Fallback for unknown labels
                    label_ids.append(-100)
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

print("Tokenizing and aligning labels...")
tokenized_datasets = dataset_split.map(tokenize_and_align_labels, batched=True)

# 5. MODEL INITIALIZATION
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)
model.to(device)

# 6. METRICS
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# 7. TRAINING ARGUMENTS (High Accuracy Mode)
args = TrainingArguments(
    output_dir="./results/local",
    eval_strategy="epoch",
    save_strategy="epoch",
    
    # 1. Increase Learning Rate slightly to help it learn faster
    learning_rate=5e-5,
    
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    
    # 2. CRITICAL CHANGE: Increase Epochs from 3 to 20
    num_train_epochs=20,
    
    weight_decay=0.01,
    logging_steps=50,
    dataloader_pin_memory=False,
    
    # 3. Only save the very best model (saves disk space)
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer,      # Updated from 'tokenizer' to fix warning
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 8. TRAIN
print("Starting training...")
trainer.train()

# 9. SAVE MODEL
print("Saving model...")
trainer.save_model("models/dholuo_pos_model")
tokenizer.save_pretrained("models/dholuo_pos_model")
print("Model saved to models/dholuo_pos_model/")