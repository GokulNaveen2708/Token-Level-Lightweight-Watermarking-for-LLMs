import json
import os

import numpy as np
import torch
import hashlib
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm

# Default watermark parameters
SPLIT_FRACTION = 0.5  # Fraction of vocabulary to bias (greenlist)
BIAS = 4.0  # Logit bias strength


def get_data_path(filename):
    """
    Construct the absolute path to a file in the `data` directory.

    Args:
        filename (str): Name of the file.

    Returns:
        str: Absolute path to the file inside the `data` directory.
    """
    root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(root, 'data', filename)


def load_prompts(path=None):
    """
    Load prompts from a JSON file.

    Args:
        path (str, optional): Path to the prompts JSON file.
                              Defaults to `reddit_prompts.json` in the `data` directory.

    Returns:
        list[dict]: A list of prompt dictionaries containing 'text' and 'prompt_id'.
    """
    if path is None:
        path = get_data_path("reddit_prompts.json")
    with open(path, "r") as f:
        return json.load(f)


def save_generation(output, path):
    """
    Save a single generation result to a JSONL file.

    Args:
        output (dict): A dictionary containing generation metadata and text.
        path (str): Path to the output JSONL file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(output) + '\n')


class SoftWatermarkLogitsProcessor(LogitsProcessor):
    """
    A custom logits processor that applies a soft watermark during text generation
    by adding a bias to a subset (greenlist) of the vocabulary. Sourced from Kirchenbauer Implementation

    The greenlist is determined by:
        - Taking the last `prefix_len` tokens
        - Combining them with a secret key
        - Hashing the result to deterministically shuffle the vocabulary
        - Selecting the top `gamma` fraction as the greenlist
    """

    def __init__(self, secret_key, vocab_size, gamma=SPLIT_FRACTION, alpha=BIAS, prefix_len=1):
        """
        Args:
            secret_key (str): Secret key used for deterministic vocabulary shuffling.
            vocab_size (int): Size of the tokenizer vocabulary.
            gamma (float): Fraction of tokens to bias (greenlist size).
            alpha (float): Logit bias value to add to greenlist tokens.
            prefix_len (int): Number of tokens from prefix used for hashing.
        """
        self.secret_key = secret_key
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.alpha = alpha
        self.prefix_len = prefix_len

    def _get_seed(self, prefix_token_ids):
        """
        Generate a deterministic random seed based on the last `prefix_len` tokens and secret key.

        Args:
            prefix_token_ids (list[int]): List of token IDs generated so far.

        Returns:
            int: Deterministic seed value.
        """
        prefix_str = "_".join(str(tok) for tok in prefix_token_ids[-self.prefix_len:])
        full_str = f"{prefix_str}_{self.secret_key}".encode("utf-8")
        return int(hashlib.sha256(full_str).hexdigest(), 16)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply the watermark bias to logits before sampling the next token.

        Args:
            input_ids (torch.LongTensor): Current generated token IDs (batch size 1).
            scores (torch.FloatTensor): Model's output logits for next token prediction.

        Returns:
            torch.FloatTensor: Adjusted logits with watermark bias applied.
        """
        prefix_token_ids = input_ids[0].tolist()
        seed = self._get_seed(prefix_token_ids)

        # Shuffle vocabulary deterministically based on seed
        rng = np.random.default_rng(seed)
        vocab_indices = list(range(self.vocab_size))
        rng.shuffle(vocab_indices)

        # Select top gamma fraction as greenlist
        greenlist_size = int(self.gamma * self.vocab_size)
        greenlist = vocab_indices[:greenlist_size]

        # Apply positive bias to greenlist tokens
        bias = torch.zeros_like(scores)
        for token_id in greenlist:
            bias[0, token_id] = self.alpha

        return scores + bias


def generate_watermarked(model, tokenizer, prompts, secret_key, output_path=None):
    """
    Generate watermarked text for given prompts using the SoftWatermarkLogitsProcessor.

    Args:
        model (transformers.PreTrainedModel): The language model for generation.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for encoding/decoding text.
        prompts (list[dict]): List of prompt dictionaries with 'text' and 'prompt_id'.
        secret_key (str): Key used to generate deterministic greenlists.
        output_path (str, optional): Path to save the generated outputs.
                                     Defaults to 'watermark_reddit.jsonl' in `data` directory.
    """
    if output_path is None:
        output_path = get_data_path("watermark_reddit.jsonl")

    model.eval()
    device = model.device
    vocab_size = tokenizer.vocab_size

    # Create logits processor for watermarking
    processor = LogitsProcessorList([
        SoftWatermarkLogitsProcessor(
            secret_key=secret_key,
            vocab_size=vocab_size,
            gamma=SPLIT_FRACTION,
            alpha=BIAS,
            prefix_len=1
        )
    ])

    for item in tqdm(prompts, desc="Generating Watermarked outputs"):
        prompt_text = item["text"]
        prompt_id = item["prompt_id"]

        # Tokenize input prompt
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        prompt_len = input_ids.shape[1]

        # Determine how many new tokens to generate
        max_tokens = min(300, 1024 - prompt_len)
        max_new_tokens = random.randint(250, max_tokens)

        # Skip very short generations
        if max_new_tokens < 10:
            continue

        # Generate output with watermarking
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            logits_processor=processor,
            do_sample=True,
            top_k=0,
            top_p=1.0,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        # Extract generated tokens (excluding prompt)
        full_output = output_ids[0]
        generated_ids = full_output[prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Save generation metadata and text
        save_generation({
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "generated_text": generated_text,
            "model": "gpt2 (watermarked)",
            "bias_strength": BIAS,
            "secret_key": secret_key,
            "prompt_tokens": prompt_len,
            "generated_tokens": len(generated_ids)
        }, path=output_path)


if __name__ == "__main__":
    # Load GPT-2 model and tokenizer
    print("Loading GPT-2 base model (Watermarked)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    # Load prompts
    print("Loading prompts")
    prompts = load_prompts()

    # Generate outputs with watermarking
    print("Generating outputs with watermarking")
    generate_watermarked(model, tokenizer, prompts, secret_key="watermark123")
    print("Watermarked Generation Complete")
