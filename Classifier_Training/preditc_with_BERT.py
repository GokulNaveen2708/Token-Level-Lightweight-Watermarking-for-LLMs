import os
import json
import torch
import pandas as pd
from datasets import Dataset
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_DIR = "model/bert_watermark_classifier_gamma=0.5+0.2/final_model"
DATASETS = {
    "Watermarked Text": "watermark.jsonl",
    "Paraphrased Text": "paraphrased_watermark_reddit.jsonl",
    "reddit_text": "watermark_reddit.jsonl",
    "watermark  γ =0.5": "watermark_split=0.5.jsonl"

}
BATCH_SIZE = 64

# Load Model + Tokenizer
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Load JSONL Data
def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def get_raw_data_path(filename):
    root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(root, 'data', filename)


def predict_batchwise(texts):
    preds = []
    for i in range(0, len(texts[:1000]), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        encoded = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            batch_preds = probs.argmax(dim=1).cpu().numpy()
            preds.extend(batch_preds)
    return preds


# === Accuracy Evaluation
accuracies = {}
for label, path in DATASETS.items():
    data = load_jsonl(get_raw_data_path(path))
    df = pd.DataFrame(data)

    # Use "text" or "generated_text" depending on the file
    text_column = "paraphrased_text" if "paraphrased_text" in df.columns else "generated_text"
    text_column = "corrupted_text" if "corrupted_text" in df.columns else "generated_text"
    texts = df[text_column].tolist()

    predictions = predict_batchwise(texts)
    correct = sum(p == 1 for p in predictions)
    total = len(predictions)
    acc = correct / total
    accuracies[label] = acc
    print(f"{label}: {acc:.2%} ({correct}/{total})")

# Plot
labels = list(accuracies.keys())
values = [v * 100 for v in accuracies.values()]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, values, color=["orange", "skyblue", "lightgreen", "purple"])
plt.ylim(0, 105)
plt.ylabel("Detection Accuracy (%)")
plt.title("BERT Trained on Watermarked Text: Watermarked vs. Paraphrased vs. Token Substitution")

for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}%", ha='center', fontsize=10)

plt.grid(axis='y', linestyle="--", alpha=0.5)
plt.tight_layout()

# Save plot
os.makedirs("Results", exist_ok=True)
plt.savefig("Plot_Results/classifier_accuracy_comparison_0.5.png", dpi=300)



