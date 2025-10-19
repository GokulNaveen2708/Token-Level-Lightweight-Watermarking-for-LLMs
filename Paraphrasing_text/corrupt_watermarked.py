import os
import json
import random
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# Config
FRACTION = 0.7
MODE = "delete"
SEED = 21
random.seed(SEED)


def get_data_path(filename):
    root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(root, "data", filename)


INPUT_JSONL = get_data_path("watermark.jsonl")
OUTPUT_JSONL = get_data_path(f"corrupted_{MODE}_{int(FRACTION * 100)}.jsonl")


def corrupt_text(text, frac, mode="substitute"):
    tokens = word_tokenize(text)
    n = len(tokens)
    num_to_modify = int(frac * n)
    indices = random.sample(range(n), min(num_to_modify, n))

    if mode == "delete":
        modified = [tok for i, tok in enumerate(tokens) if i not in indices]
    elif mode == "substitute":
        modified = [tok if i not in indices else "[MASK]" for i, tok in enumerate(tokens)]
    else:
        raise ValueError("Unknown mode:", mode)

    return " ".join(modified)


# Load data
with open(INPUT_JSONL) as f:
    entries = [json.loads(line) for line in f]

# Apply corruption
with open(OUTPUT_JSONL, "w") as fout:
    for entry in tqdm(entries[:1000], desc=f"Applying {MODE} corruption"):
        original = entry["generated_text"]
        corrupted = corrupt_text(original, FRACTION, mode=MODE)
        entry["corrupted_text"] = corrupted
        fout.write(json.dumps(entry) + "\n")

print(f"Corrupted data saved to: {OUTPUT_JSONL}")
