import json
import os
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def get_data_path(filename):
    root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(root, "data", filename)


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def analyze_lengths(generations):
    prompt_lens = [g["prompt_tokens"] for g in generations]
    gen_lens = [g["generated_tokens"] for g in generations]
    total_lens = [p + g for p, g in zip(prompt_lens, gen_lens)]

    print(f" Average prompt length: {sum(prompt_lens) / len(prompt_lens):.2f} tokens")
    print(f" Average generation length: {sum(gen_lens) / len(gen_lens):.2f} tokens")
    print(f" Max total token length: {max(total_lens)}")

    return prompt_lens, gen_lens, total_lens


def plot_histogram(data, title, xlabel, ylabel, bins=30):
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=bins, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def preview_samples(generations, n=5):
    print("\n Sample Generations:")
    for g in generations[:n]:
        print(f"\nPrompt ID: {g['prompt_id']}")
        print(f"Prompt:\n{g['prompt']}")
        print(f"Generated:\n{g['generated_text']}")
        print("-" * 40)


if __name__ == "__main__":
    path = get_data_path("no_watermark.jsonl")
    generations = load_jsonl(path)

    prompt_lens, gen_lens, total_lens = analyze_lengths(generations)

    # Visualize
    plot_histogram(prompt_lens, "Prompt Token Lengths", "Tokens", "Count")
    plot_histogram(gen_lens, "Generated Token Lengths", "Tokens", "Count")
    plot_histogram(total_lens, "Total Lengths (Prompt + Generation)", "Tokens", "Count")

    # Sample Output
    preview_samples(generations)
