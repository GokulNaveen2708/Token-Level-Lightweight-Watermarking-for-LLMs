import os, json, math, torch, hashlib
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binomtest
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

# Constants
MODEL_ID = "gpt2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SECRET_KEY = "watermark123"
SPLIT_FRACTION = 0.5
PREFIX_LEN = 1

# Load model/tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()
VOCAB_SIZE = tokenizer.vocab_size


# Paths
def get_data_path(filename):
    root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(root, "data", filename)


def get_results_path(filename):
    root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(root, "Plot_Results", filename)


# Hash & Greenlist
def hash_seed(prefix_ids, key):
    prefix_str = "_".join(str(t) for t in prefix_ids).encode()
    seed_str = prefix_str + f"_{key}".encode()
    return int(hashlib.sha256(seed_str).hexdigest(), 16)


def get_greenlist(seed, split=SPLIT_FRACTION):
    rng = np.random.default_rng(seed)
    vocab_indices = list(range(VOCAB_SIZE))
    rng.shuffle(vocab_indices)
    return set(vocab_indices[:int(split * VOCAB_SIZE)])


# Core Metrics
def calc_z_score(green, total, p=SPLIT_FRACTION):
    expected = total * p
    std = math.sqrt(total * p * (1 - p))
    return (green - expected) / std if std != 0 else 0


def binomial_z_score(green, total, p=SPLIT_FRACTION):
    if total == 0:
        return 0.0
    test = binomtest(green, total, p, alternative='greater')
    p_val = test.pvalue
    return -np.log10(p_val) if p_val > 0 else 10.0  # Cap large sc


def perplexity(text):
    ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        loss = model(ids, labels=ids).loss
    return math.exp(loss.item())


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


# Evaluation
def evaluate(wm_data, unwm_data):
    position_stats_global = defaultdict(lambda: [0, 0])  # Global stats for heatmap
    results = []


    wm_dict = {x["prompt_id"]: x for x in wm_data}
    unwm_dict = {x["prompt_id"]: x for x in unwm_data}
    common_ids = sorted(set(wm_dict.keys()) & set(unwm_dict.keys()))
    missing = set(wm_dict.keys()) ^ set(unwm_dict.keys())

    if missing:
        print(f" Warning: {len(missing)} prompt skipped due to missing counterpart.")

    for pid in tqdm(common_ids[:1000], desc="Evaluating prompts"):
        wm = wm_dict[pid]
        unwm = unwm_dict[pid]

        prompt = wm["prompt"]
        wm_text = wm["generated_text"]
        unwm_text = unwm["generated_text"]

        full_wm = tokenizer(prompt + " " + wm_text, add_special_tokens=False).input_ids
        wm_tokens = tokenizer(wm_text, add_special_tokens=False).input_ids
        unwm_tokens = tokenizer(unwm_text, add_special_tokens=False).input_ids
        start = len(full_wm) - len(wm_tokens)

        # Watermarked
        green = 0
        for i in range(len(wm_tokens)):
            prefix_start = start + i - PREFIX_LEN
            prefix_tokens = full_wm[prefix_start:start + i] if prefix_start >= 0 else full_wm[:start + i]
            seed = hash_seed(prefix_tokens, SECRET_KEY)
            greenlist = get_greenlist(seed)
            tok = wm_tokens[i]

            position_stats_global[i][1] += 1
            if tok in greenlist:
                green += 1
                position_stats_global[i][0] += 1

        wm_z = calc_z_score(green, len(wm_tokens))

        wm_green_rate = green / len(wm_tokens) if wm_tokens else 0

        # Unwatermarked
        green_unwm = 0
        for i in range(len(unwm_tokens)):
            prefix_start = i - PREFIX_LEN
            prefix_tokens = unwm_tokens[prefix_start:i] if prefix_start >= 0 else full_wm[:start + i]
            seed = hash_seed(prefix_tokens, SECRET_KEY)
            greenlist = get_greenlist(seed)
            tok = unwm_tokens[i]
            if tok in greenlist:
                green_unwm += 1

        unwm_z = calc_z_score(green_unwm, len(unwm_tokens))
        unwm_green_rate = green_unwm / len(unwm_tokens) if unwm_tokens else 0
        results.append({
            "prompt_id": pid,
            "watermarked_z_score": wm_z,
            "unwatermarked_z_score": unwm_z,
            "watermarked_ppl": perplexity(prompt + " " + wm_text),
            "no_watermark_ppl": perplexity(prompt + " " + unwm_text),
            "green_tokens": green,
            "total_tokens": len(wm_tokens),
            "wm_green_rate": wm_green_rate,
            "unwm_green_rate": unwm_green_rate
        })

    return results, position_stats_global


def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")


def plot_green_fraction(stats):
    import matplotlib
    matplotlib.use("Agg")
    positions = sorted(stats)
    fractions = [stats[p][0] / stats[p][1] for p in positions]
    plt.figure(figsize=(10, 4))
    sns.lineplot(x=positions, y=fractions)
    plt.title("Green Token Fraction by Position")
    plt.xlabel("Token Position")
    plt.ylabel("Fraction Green")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("Plot_Results", exist_ok=True)
    plt.savefig("Plot_Results/green_token_fraction_reddit.png")
    print(" Plot saved to Plot_Results/green_token_fraction_corrupted_3%.png")


def plot_zscore_histogram(results, save_path="Plot_Results/zscore_histogram_poster.png"):
    import matplotlib
    matplotlib.use("Agg")  # Use non-GUI backend for saving

    # Extract Z-scores
    wm_z = [entry["watermarked_z_score"] for entry in results]
    unwm_z = [entry["unwatermarked_z_score"] for entry in results]

    # Create plot
    plt.figure(figsize=(8, 4))  # Optimized for poster/slide size
    sns.histplot(wm_z, bins=30, kde=True, color="green", label="Watermarked", stat="count")
    sns.histplot(unwm_z, bins=30, kde=True, color="red", label="Unwatermarked", stat="count")

    # Formatting
    plt.axvline(0, color='gray', linestyle='--')
    # plt.title("Z-score Distribution: Watermarked vs Unwatermarked", fontsize=14, fontweight='bold')
    plt.xlabel("Z-score", fontsize=12, fontweight='bold')
    plt.ylabel("Frequency", fontsize=12, fontweight='bold')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(
        fontsize=18,
        loc='upper right',
        frameon=True,
        framealpha=1.0
    )
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Z-score histogram saved to {save_path}")


def plot_perplexity_shift(wm_path, unwm_path, save_path="Plot_Results/perplexity_shift_poster.png"):
    import numpy as np
    wm_data = load_jsonl(wm_path)
    unwm_data = load_jsonl(unwm_path)

    wm_dict = {x["prompt_id"]: x for x in wm_data}
    unwm_dict = {x["prompt_id"]: x for x in unwm_data}
    shared_ids = sorted(set(wm_dict) & set(unwm_dict))

    deltas = [wm_dict[pid]["watermarked_ppl"] - unwm_dict[pid]["no_watermark_ppl"] for pid in shared_ids]
    clipped_deltas = np.clip(deltas, -100, 100)

    plt.figure(figsize=(10, 6))
    # sns.set(style="whitegrid", font_scale=1.1)
    sns.histplot(clipped_deltas, bins=np.arange(-100, 105, 5), kde=True, color="purple", edgecolor='black')

    # plt.axvline(0, linestyle="--", color="black", linewidth=1.2, label="No Shift")
    # plt.title("Perplexity Shift due to Watermarking", fontsize=18, fontweight='bold')
    plt.xlabel("Δ Perplexity (Watermarked - Unwatermarked)", fontsize=13, fontweight='bold')
    plt.ylabel("Number of Prompts", fontsize=13, fontweight='bold')
    # # plt.legend(
    #     fontsize=18,
    #     loc='upper right',
    #     frameon=True,
    #     framealpha=1.0
    # )
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path,dpi = 300)
    print(f" Saved plot to {save_path}")


def run_watermark_diagnostics(result_path="evaluation_paraphrased_reddit.jsonl", save_plot_path="Plot_Results/z_vs_ppl_delta_reddit.png"):
    """
    Analyze z-score performance and perplexity delta from evaluation results.
    Outputs detection stats and saves a diagnostic scatter plot.

    Args:
        result_path (str): Path to the .jsonl file containing evaluation results.
        save_plot_path (str): Path to save the z-score vs perplexity delta plot.
    """

    # Load results
    results = load_jsonl(get_results_path(result_path))
    if not results:
        print(" No data found.")
        return

    # Extract metrics
    wm_z_scores = [r["watermarked_z_score"] for r in results]
    ppl_deltas = [r["watermarked_ppl"] - r["no_watermark_ppl"] for r in results]

    # Compute statistics
    num_detected = sum(z > 4.0 for z in wm_z_scores)
    detection_rate = num_detected / len(wm_z_scores)
    avg_ppl_change = sum(ppl_deltas) / len(ppl_deltas)

    print(f" Detection Success Rate (binomial score > 4): {detection_rate:.2%}")
    print(f" Avg Δ Perplexity (WM - UnWM): {avg_ppl_change:.2f}")

    # Plot
    os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.scatter(ppl_deltas, wm_z_scores, alpha=0.7)
    plt.axhline(1, color='blue', linestyle='--', label='Z = 1 threshold')
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel("Δ Perplexity (WM - UnWM)")
    plt.ylabel("Watermarked Detection Score")
    plt.title("Z-score vs. Perplexity Delta")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_plot_path)
    print(f" Plot saved to {save_plot_path}")


def run_empirical_threshold_diagnostics(
        result_path="evaluation_reddit.jsonl",
        save_plot_path="Plot_Results/zscore_empirical_comparison_reddit.png",
        std_multiplier=3
):
    """
    Computes detection threshold using empirical unwatermarked z-score stats.
    Reports detection accuracy based on that.

    Args:
        result_path: Path to jsonl file with evaluation results.
        save_plot_path: Where to save diagnostic plot.
        std_multiplier: How many stds above unwm mean to set threshold.
    """
    results = load_jsonl(get_results_path(result_path))
    if not results:
        print(" No data found.")
        return

    wm_z = [r["watermarked_z_score"] for r in results]
    unwm_z = [r["unwatermarked_z_score"] for r in results]

    mu = np.mean(unwm_z)
    sigma = np.std(unwm_z)
    threshold = mu + std_multiplier * sigma

    detected = sum(z > threshold for z in wm_z)
    total = len(wm_z)
    rate = detected / total

    print(f"Empirical mean(unwm) = {mu:.2f}, std = {sigma:.2f}")
    print(f"Detection threshold = {threshold:.2f} (μ + {std_multiplier}σ)")
    print(f" Detected {detected} / {total} → Detection rate = {rate:.2%}")

    # Plot
    os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    sns.histplot(wm_z, color="green", label="Watermarked", bins=30, kde=True, stat="density")
    sns.histplot(unwm_z, color="red", label="Unwatermarked", bins=30, kde=True, stat="density")
    plt.axvline(threshold, color='blue', linestyle='--', label=f"μ+{std_multiplier}σ")
    plt.title("Z-score Distribution with Empirical Threshold")
    plt.xlabel("Z-score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_plot_path)
    print(f" Plot saved to {save_plot_path}")


def plot_roc_auc(results_path="evaluation_reddit.jsonl", save_path="Plot_Results/roc_auc_curve_reddit.png"):
    results = load_jsonl(get_results_path(results_path))
    if not results:
        print("No results found.")
        return

    y_true = []
    y_scores = []

    for r in results:
        y_true.append(1)
        y_scores.append(r["watermarked_z_score"])

        y_true.append(0)
        y_scores.append(r["unwatermarked_z_score"])

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Z-Score Watermark Detection")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"ROC AUC curve saved to {save_path}")


def plot_precision_recall(results, save_path="Plot_Results/precision_recall_curve_reddit.png"):
    labels = [1] * len(results)
    scores = [r["watermarked_z_score"] for r in results]
    unwm_scores = [r["unwatermarked_z_score"] for r in results]
    labels += [0] * len(unwm_scores)
    scores += unwm_scores

    precision, recall, _ = precision_recall_curve(labels, scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="green", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Precision Recall plot saved to {save_path}")


def plot_confusion_matrix(results, threshold=4.0, save_path="Plot_Results/confusion_matrix_reddit.png"):
    labels = [1] * len(results)
    scores = [r["watermarked_z_score"] for r in results]
    unwm_scores = [r["unwatermarked_z_score"] for r in results]
    labels += [0] * len(unwm_scores)
    scores += unwm_scores

    preds = [1 if score > threshold else 0 for score in scores]
    cm = confusion_matrix(labels, preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["UnWM", "WM"])
    plt.figure(figsize=(6, 5))
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix (Threshold={threshold})")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")


# Run
if __name__ == "__main__":
    wm_data = load_jsonl(get_data_path("paraphrased_watermark_reddit.jsonl"))
    unwm_data = load_jsonl(get_data_path("no_watermark_reddit.jsonl"))
    results, stats = evaluate(wm_data, unwm_data)
    save_results(results, get_results_path("evaluation_paraphrased_reddit.jsonl"))
    plot_green_fraction(stats)
    #
    plot_perplexity_shift(get_results_path("evaluation_results_gamma=0.5.jsonl"), get_results_path("evaluation_results_gamma=0.5.jsonl"))
    run_watermark_diagnostics()
    run_empirical_threshold_diagnostics()
    plot_roc_auc()
    results = load_jsonl(get_results_path("evaluation_results2.jsonl"))
    plot_zscore_histogram(results)
    plot_precision_recall(results)
    plot_confusion_matrix(results)

    print(" Evaluation complete. Plot_Results saved.")
