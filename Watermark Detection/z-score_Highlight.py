import hashlib
import numpy as np
from transformers import AutoTokenizer

# === Config
MODEL_ID = "gpt2"
SECRET_KEY = "watermark123"
SPLIT_FRACTION = 0.5
PREFIX_LEN = 1

# === Metrics (from evaluation_results2.jsonl)
metrics = {
    "z_score_wm": 27.44,
    "z_score_unwm": 1.09,
    "green_rate_wm": 0.8401,
    "green_rate_unwm": 0.2312,
    "ppl_wm": 15.70,
    "ppl_unwm": 12.80
}

# === Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
VOCAB_SIZE = tokenizer.vocab_size

def hash_seed(prefix_ids, key):
    prefix_str = "_".join(str(t) for t in prefix_ids).encode()
    seed_str = prefix_str + f"_{key}".encode()
    return int(hashlib.sha256(seed_str).hexdigest(), 16)

def get_greenlist(seed, split=SPLIT_FRACTION):
    rng = np.random.default_rng(seed)
    vocab_indices = list(range(VOCAB_SIZE))
    rng.shuffle(vocab_indices)
    return set(vocab_indices[:int(split * VOCAB_SIZE)])

def highlight_text_colored(prompt, generated, key=SECRET_KEY):
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    gen_enc = tokenizer(generated, add_special_tokens=False, return_tensors="pt")
    gen_ids = gen_enc.input_ids[0].tolist()
    full_ids = prompt_ids + gen_ids

    html_tokens = []
    for i, token_id in enumerate(gen_ids):
        prefix_start = len(prompt_ids) + i - PREFIX_LEN
        prefix_tokens = full_ids[prefix_start:len(prompt_ids) + i] if prefix_start >= 0 else full_ids[:len(prompt_ids) + i]
        seed = hash_seed(prefix_tokens, key)
        greenlist = get_greenlist(seed)
        token_str = tokenizer.decode([token_id])
        token_str = token_str.replace("<", "&lt;").replace(">", "&gt;")
        if token_id in greenlist:
            html_tokens.append(f'<span style="background-color:#d4edda">{token_str}</span>')
        else:
            html_tokens.append(f'<span style="background-color:#f8d7da">{token_str}</span>')
    return "".join(html_tokens)
# === Inputs
prompt = (
    "Sysco Corp. has terminated its planned $3.5 billion takeover of US Foods, it announced Monday, after a federal judge "
    "blocked the combination. The company is opting instead to add $3 billion to its stock-buyback program."
)

watermarked_text = (
    " New Jersey State Representative R. J. Yaine, who also has a health insurance mandate, opposed the deal. "
    "New Jersey Senate President Pro Tem Keith Henschel opposed it. Sales and marketing chairman Steve Murphy called "
    "the decision a key victory."
)

unwatermarked_text = (
    "The company's stock trades on Nasdaq at $45.29 per share. Sysco has been the second biggest provider to McDonald's "
    "for more than 80 years. The company added 13 stores in six states recently."
)

# Generate colored text
highlighted_wm = highlight_text_colored(prompt, watermarked_text)
highlighted_unwm = highlight_text_colored(prompt, unwatermarked_text)

# === HTML Layout (Vertical Format)
html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Vertical Watermark Table</title>
    <style>
        table {{
            border-collapse: collapse;
            width: 100%;
            font-family: Arial, sans-serif;
            font-size: 15px;
        }}
        th, td {{
            border: 1px solid #ccc;
            padding: 10px;
            vertical-align: top;
        }}
        th {{
            background-color: #f2f2f2;
            text-align: left;
        }}
        .label-cell {{
            font-weight: bold;
            width: 15%;
            background-color: #f9f9f9;
        }}
    </style>
</head>
<body>
    <h2>Token-Level Watermark Detection</h2>
    <table>
    <tr>
        <th class="label-cell">Type</th>
        <th>Text</th>
        <th>Z-Score</th>
        <th>Perplexity</th>
    </tr>
    <tr>
        <td><strong>Prompt</strong></td>
        <td>{prompt}</td>
        <td>—</td>
        <td>—</td>
    </tr>
    <tr>
        <td><strong>Unwatermarked Output</strong></td>
        <td>{highlighted_unwm}</td>
        <td>{metrics['z_score_unwm']:.2f}</td>
        <td>{metrics['ppl_unwm']:.2f}</td>
    </tr>
    <tr>
        <td><strong>Watermarked Output</strong></td>
        <td>{highlighted_wm}</td>
        <td>{metrics['z_score_wm']:.2f}</td>
        <td>{metrics['ppl_wm']:.2f}</td>
    </tr>
</table>

</body>
</html>
"""

# === Save HTML
with open("highlight_comparison.html", "w") as f:
    f.write(html)

print("Vertical HTML saved: vertical_watermark_table.html")