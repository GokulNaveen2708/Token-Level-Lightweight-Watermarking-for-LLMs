import os
import json
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Configuration
MODEL_NAME = "Vamsi/T5_Paraphrase_Paws"
DEVICE ="cpu"


def get_data_path(filename):
    """
    Get the absolute path to a file inside the 'data' directory.

    Args:
        filename (str): The file name.

    Returns:
        str: Absolute path to the file in the 'data' folder.
    """
    root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(root, "data", filename)


def get_results_path(filename):
    """
    Get the absolute path to a file inside the 'Plot_Results' directory.

    Args:
        filename (str): The file name.

    Returns:
        str: Absolute path to the file in the 'Plot_Results' folder.
    """
    root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(root, "Plot_Results", filename)



# File Paths

INPUT_JSONL = get_data_path("watermark_reddit.jsonl")  # Input: watermarked generations
OUTPUT_JSONL = get_data_path("paraphrased_watermark_reddit.jsonl")  # Output: paraphrased results


# Load paraphrasing model

print("Loading paraphrasing model")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


def paraphrase(text, max_len=350):
    """
    Generate a paraphrased version of the given text using the T5 model.

    Args:
        text (str): Input text to paraphrase.
        max_len (int): Maximum length of the paraphrased output.

    Returns:
        str: Paraphrased text.
    """
    # Prefix input with "paraphrase:" as required by the model
    input_text = f"paraphrase: {text} </s>"

    # Tokenize input
    inputs = tokenizer([input_text], return_tensors="pt", truncation=True, padding="longest").to(DEVICE)

    # Generate paraphrase using beam search
    outputs = model.generate(
        **inputs,
        max_length=max_len,
        num_beams=5,
        early_stopping=True,
        num_return_sequences=1
    )

    # Decode output tokens into text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Load watermark data

with open(INPUT_JSONL) as f:
    entries = [json.loads(line) for line in f]

# Run paraphrasing

print(f" Paraphrasing {len(entries)} entries...")
os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

with open(OUTPUT_JSONL, "w") as fout:
    for entry in tqdm(entries[:1000], desc="Paraphrasing"):
        gen_text = entry["generated_text"]
        try:
            # Generate paraphrase
            para = paraphrase(gen_text)

            # Store paraphrased text in entry
            entry["paraphrased_text"] = para

            # Save to output JSONL
            fout.write(json.dumps(entry) + "\n")
        except Exception as e:
            # Log failures for specific prompts
            print(f"Failed for prompt_id={entry['prompt_id']}: {e}")

print(f" Paraphrased data saved to: {OUTPUT_JSONL}")
