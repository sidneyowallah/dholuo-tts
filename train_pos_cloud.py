import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import numpy as np

# 1. Configuration
# Note: afro-xlmr-large-76L explicitly includes Dholuo (Luo) in its pre-training
model_name = "Davlan/afro-xlmr-large-76L" 
dataset_path = "data/dho.parquet"

# 2. Load Dataset
dataset = load_dataset("parquet", data_files=dataset_path)

# Group tokens by sentence
def group_by_sentence(examples):
    sentences = {}
    for token, pos, sent_id in zip(examples["token"], examples["pos_tag"], examples["sentence_id"]):
        if sent_id not in sentences:
            sentences[sent_id] = {"tokens": [], "pos_tags": []}
        sentences[sent_id]["tokens"].append(token)
        sentences[sent_id]["pos_tags"].append(pos)
    return {"tokens": [s["tokens"] for s in sentences.values()], "pos_tags": [s["pos_tags"] for s in sentences.values()]}

grouped = dataset["train"].map(group_by_sentence, batched=True, remove_columns=dataset["train"].column_names)

# Get unique POS tags and create label mappings
all_pos_tags = set()
for tags in grouped["pos_tags"]:
    all_pos_tags.update(tags)
label2id = {tag: i for i, tag in enumerate(sorted(all_pos_tags))}
id2label = {i: tag for tag, i in label2id.items()}
num_labels = len(label2id)

dataset = grouped.train_test_split(test_size=0.1)

# 3. Tokenization
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    tokenized = tokenizer(examples["tokens"], padding="max_length", truncation=True, max_length=128, is_split_into_words=True)
    labels = []
    for i, tags in enumerate(examples["pos_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = [-100 if word_id is None else label2id[tags[word_id]] for word_id in word_ids]
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 4. Load Model
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels, id2label=id2label, label2id=label2id) 

# 5. Training Arguments (RunPod Optimized)
training_args = TrainingArguments(
    output_dir="./results/cloud",
    num_train_epochs=3,
    per_device_train_batch_size=8,   # 3090/4090 can handle 8-16
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,   # Simulates a batch size of 16
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    fp16=True,                       # Critical for speed and memory on RunPod
    gradient_checkpointing=True,     # Saves massive VRAM on 'Large' models
    load_best_model_at_end=True,
)

# 6. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# 7. Start Training
print("Starting training...")
trainer.train()

# 8. Save Model
print("Saving model...")
trainer.save_model("./dholuo_pos_cloud_model")
tokenizer.save_pretrained("./dholuo_pos_cloud_model")
print("Model saved to dholuo_pos_cloud_model/")