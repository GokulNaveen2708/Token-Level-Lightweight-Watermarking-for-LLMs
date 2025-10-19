import os
import json
import pandas as pd
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import torch

# === Config ===
MODEL_NAME = "bert-base-uncased"
DATA_PATH = "human_vs_model_dataset_gamma_0.5.jsonl"
OUTPUT_DIR = "model/bert_watermark_classifier_gamma=0.5+0.2"


# === Load JSONL Dataset ===
def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def get_data_path(filename):
    root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(root, "classification_data", filename)


raw_data = load_jsonl(get_data_path(DATA_PATH))
df = pd.DataFrame(raw_data)

print(f"Label distribution:\n{df['label'].value_counts()}")

# Tokenization
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )


# Dataset Format 
dataset = Dataset.from_pandas(df[["text", "label"]])
dataset = dataset.map(tokenize_fn, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

#  Train/Test Split
split = dataset.train_test_split(test_size=0.3, seed=42)
train_ds = split["train"]
eval_ds = split["test"]

# Model 
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Training Config 
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    num_train_epochs=1,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)




def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    try:
        auc = roc_auc_score(labels, pred.predictions[:, 1])
    except:
        auc = -1.0  # fallback if only one class
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': auc
    }


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=compute_metrics
)
# Train
trainer.train()
# Evaluate
metrics = trainer.evaluate()
print(" Final Evaluation Metrics:")
print(metrics)

# Save model
print("Saving model and tokenizer...")
model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
print("Model saved at:", os.path.join(OUTPUT_DIR, "final_model"))
