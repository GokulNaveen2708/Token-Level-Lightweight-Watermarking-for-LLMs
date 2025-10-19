import json
import os
import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm


def get_data_path(filename):
    """
    Construct the absolute path to a file in the `data` directory.

    Args:
        filename (str): Name of the file.

    Returns:
        str: Absolute path to the file inside the `data` directory.
    """
    root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(root, "data", filename)


def load_prompts(path=None):
    """
    Load prompts from a JSON file.

    Args:
        path (str, optional): Path to the prompts JSON file.
                              If None, defaults to `reddit_prompts.json` in the `data` directory.

    Returns:
        list[dict]: A list of prompts, where each prompt is a dictionary containing `text` and `prompt_id`.
    """
    if path is None:
        path = get_data_path("reddit_prompts.json")
    with open(path, "r") as f:
        return json.load(f)


def save_generation(output, path):
    """
    Save a single generation result to a JSONL file.

    Args:
        output (dict): A dictionary containing the generation metadata and text.
        path (str): Path to the output JSONL file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(output) + "\n")


def generate_no_watermark(model, tokenizer, prompts, output_path=None):
    """
    Generate text completions for given prompts without watermarking.

    Args:
        model (transformers.PreTrainedModel): The language model to generate text.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer corresponding to the model.
        prompts (list[dict]): List of prompt dictionaries containing 'text' and 'prompt_id'.
        output_path (str, optional): Path to save the generated outputs.
                                     Defaults to 'no_watermark_reddit.jsonl' in the `data` directory.
    """
    if output_path is None:
        output_path = get_data_path("no_watermark_reddit.jsonl")

    for sentence in tqdm(prompts, desc="Generating (No Watermark)"):
        prompt_text = sentence["text"]
        prompt_id = sentence["prompt_id"]

        # Tokenize the input prompt and move to device
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
        prompt_len = input_ids.shape[1]

        # Determine max number of tokens to generate
        max_tokens = min(300, 1024 - prompt_len)
        max_new_tokens = random.randint(250, max_tokens)

        # Skip very short prompts
        if max_new_tokens < 10:
            continue

        # Generate output tokens without watermarking
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=0,
            top_p=0.95,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        # Extract only the generated portion (excluding the prompt)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        output_ids_flat = output_ids[0].tolist()
        generated_token_ids = output_ids_flat[len(prompt_ids):]

        # Decode the generated text
        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

        # Save the generation result
        save_generation({
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "generated_text": generated_text,
            "model": "gpt2",
            "prompt_tokens": prompt_len,
            "generated_tokens": len(output_ids[0]) - prompt_len
        }, path=output_path)


if __name__ == "__main__":
    # Load model and tokenizer
    print("Loading GPT-2 base model (Watermarked)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    # Load prompts
    print("Loading prompts")
    prompts = load_prompts()

    # Generate and save outputs without watermark
    print("Generating outputs with watermarking")
    generate_no_watermark(model, tokenizer, prompts)
    print("Watermarked Generation Complete ")
