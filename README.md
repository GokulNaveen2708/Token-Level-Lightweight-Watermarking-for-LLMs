# A Lightweight Watermarking Method for Large Language Models

<p align="center">
  <img src="Watermark%20Detection/Plot_Results/zscore_histogram_grouped.png" alt="Z-score distribution comparison between unwatermarked and watermarked text" width="80%"/>
  <br/>
  <em>Representative z-score separation for unwatermarked vs. watermarked generations (GPT-2, γ=0.5).</em>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#-license)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Transformers](https://img.shields.io/badge/🤗-Transformers-ff69b4)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-informational)

> **TL;DR**
> Watermark text **during generation** by **subtly biasing** token selection (keyed green list favored, red list disfavored) using a **secret key** and a **deterministic per-step vocabulary permutation**. Later, a simple **one-sided z-test** detects the hidden signal with high confidence—while keeping the text fluent and natural.

---

## ✨ Overview

Modern LLMs generate vast amounts of content, making **provenance** and **authenticity** critical for safety, moderation, and trust. This repository contains the **official implementation** of our paper, **“A Lightweight Watermarking Method for Large Language Models.”** Our method is designed to be:

- **Lightweight & Production-Ready:** Pure **inference-time** logit adjustment—**no model fine-tuning** or architectural changes.
- **Accurate & Calibratable:** A keyed **z-score detector** provides strong separation at typical lengths; thresholds (e.g., *z* ≥ 4) can be tuned for desired false-positive rates.
- **Quality-Preserving:** With small bias values (e.g., **δ = 0.5–4.0**), measured **perplexity shifts stay modest**, keeping outputs natural.
- **Robust in Practice:** Maintains useful signal under **paraphrasing**, **token substitutions**, and **minor edits**; an optional **BERT classifier** supports black-box detection when keys aren’t available.

**How it works (at a glance):**
- At each decoding step, a **secret key** and a **rolling prefix window** seed a PRNG that **permutes the vocabulary**.
- The top **γ fraction** of the permuted vocab becomes the **green list** (favored), and the rest is **red** (disfavored).
- We add a small **δ** logit bonus to green tokens before sampling.
- For detection, we **recompute** the per-step green lists with the **same key** and run a **one-sided z-test** on how many realized tokens fell into green.

**What’s in this repo:**
- A plug-and-play **watermark logits processor** for Hugging Face causal LMs (e.g., GPT-2).
- A keyed **z-score detector** and an optional **BERT-based classifier** for black-box detection.
- Scripts & notebooks for **generation**, **detection**, **robustness** experiments, and **quality** analysis.
- Ready-to-use result figures under `Watermark Detection/Plot_Results/` and `Classifier_Training/Results/` to showcase performance (z-score distributions, length scaling, confusion matrix, perplexity shifts, robustness, classifier accuracy).

---

## 🧠 Method (High Level)

This method embeds a **statistical watermark** during generation by **gently nudging** the model toward a secret, per-step **green list** of tokens. The nudging is tiny (logit bonus **δ**) and keyed, so the pattern is **invisible without the key** but **recoverable** by recomputing the same green lists at detection time.

### Core Idea

At each decoding step *t*, we derive a **seed** from the last *k* tokens (the **rolling prefix**) and a **secret key**. Using this seed, we generate a deterministic **pseudorandom ranking** over the vocabulary and choose the top **γ** fraction as the **green list** \(G_t\). We then **add δ** to logits of tokens in \(G_t\) before sampling.

- **Rolling Prefix:** the last *k* generated tokens (including prompt context if needed).
- **Secret Key:** any secret string/bytes/number known to the generator & detector.
- **γ (gamma):** green-list ratio (e.g., 0.5).
- **δ (delta):** logit bonus applied to green tokens (e.g., 0.5–4.0).
- **Deterministic PRNG:** ensures the same prefix + key recreates the same \(G_t\).

> Intuition: each position has its **own** secret green set. When the model favors these sets *slightly* over time, the final text contains **more green tokens** than chance would allow, which a detector can measure statistically.

---

### Step-by-Step

1. **Compute Seed:**
   \( \text{seed}_t = H(\text{prefix}_{t,k} \,\Vert\, \text{key}) \)
   where \(H\) is a cryptographic/non-cryptographic hash (implementation detail), \(\text{prefix}_{t,k}\) are the last \(k\) tokens, and \(\Vert\) denotes concatenation.

2. **Rank Vocabulary (Deterministic):**
   Use the PRNG seeded by \(\text{seed}_t\) to assign each vocab id a **score**.
   - Efficient option: **hash-per-token** score \( s(v) = \text{Hash}(v, \text{seed}_t) \in [0,1) \) (no full permutation needed).
   - Define \(G_t = \{v \mid s(v) < \gamma\}\).

3. **Bias Logits:**
   For tokens in \(G_t\), add **δ** to their logits:
   \[
   \ell'_v = \ell_v + \begin{cases}
   \delta & \text{if } v \in G_t\\
   0 & \text{otherwise}
   \end{cases}
   \]

4. **Sample Next Token:**
   Sample with your usual decoding (e.g., top-k, temperature, nucleus). Append to the prefix and continue.

5. **Repeat:**
   Update the rolling prefix and continue until the sequence is complete.

---

### Pseudocode (Generation)

```python
def stepwise_green_set(prefix_ids, key, gamma):
    seed = H(prefix_ids[-k:], key)          # rolling prefix + secret key
    def is_green(token_id):
        s = hash01(token_id, seed)          # in [0,1)
        return s < gamma
    return is_green

def generate(model, tokenizer, prompt, key, gamma=0.5, delta=0.5, k=3,
             max_new_tokens=200, temperature=0.8, top_k=50):
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    for _ in range(max_new_tokens):
        logits = model(ids).logits[:, -1, :]          # last step logits

        is_green = stepwise_green_set(ids[0].tolist(), key, gamma)
        green_mask = vectorize_vocab_mask(is_green, vocab_size=logits.size(-1))
        logits = logits + delta * green_mask          # apply bias

        next_id = sample(logits, temperature=temperature, top_k=top_k)
        ids = torch.cat([ids, next_id], dim=1)
    return tokenizer.decode(ids[0], skip_special_tokens=True)
```

## 🧠 Theory & Detection

This section formalizes the detection problem and provides practical guidance for **thresholds**, **p-values**, **short-text handling**, and **sliding-window** detection.

---

### Problem Setup

Consider a generated token sequence \(x_{1:n}\) with tokenizer \(T\) and vocabulary \(\mathcal{V}\).
At each step \(t\), the generator (with **secret key** \(K\)) defines a **green set** \(G_t \subset \mathcal{V}\) by seeding a deterministic PRNG with the rolling prefix and \(K\), and including the top \(\gamma\) fraction of tokens.

Define indicator variables
\[
I_t = \mathbb{1}\{x_t \in G_t\}, \quad S_n = \sum_{t=1}^n I_t.
\]

**Null hypothesis \(H_0\)** (no watermark): tokens are sampled without the keyed bias; marginally,
\[
I_t \sim \text{Bernoulli}(\gamma) \quad \Rightarrow \quad \mathbb{E}[S_n] = \gamma n,\ \ \text{Var}(S_n) \approx n\gamma(1-\gamma).
\]

**Alternative \(H_1\)** (watermarked): green tokens receive a +\(\delta\) logit bonus,
so the realized green rate \(p_\text{wm} = \Pr(x_t \in G_t)\) becomes \(> \gamma\), hence \(S_n\) tends to be larger.

> **Key intuition:** under \(H_1\), the count of “token-in-green” events accumulates **above chance**.

---

### One-Sided z-Test (CLT Approximation)

We use a standardized statistic:
\[
z = \frac{S_n - \gamma n}{\sqrt{n\,\gamma(1-\gamma)}}.
\]
Under \(H_0\) (for moderate \(n\)), \(z \approx \mathcal{N}(0,1)\) by the Central Limit Theorem.

- **Decision rule:** declare “watermarked” if \(z \ge z_\tau\).
- **p-value:** \(p = 1 - \Phi(z)\) (one-sided).

#### Choosing a Threshold
- \(z_\tau = 3.0 \Rightarrow \alpha \approx 1.35 \times 10^{-3}\)
- \(z_\tau = 4.0 \Rightarrow \alpha \approx 3.17 \times 10^{-5}\)
- \(z_\tau = 5.0 \Rightarrow \alpha \approx 2.87 \times 10^{-7}\)

> In practice, **calibrate** \(z_\tau\) to hit your target **FPR** on a clean **human** validation set.

---

### Exact (Non-Asymptotic) Binomial Test (Recommended for Short Texts)

For short sequences (\(n \lesssim 80\)), the normal approximation can be conservative or unstable.
Use the exact binomial tail:
\[
p_{\text{exact}} = \Pr_{X \sim \text{Binom}(n,\gamma)}(X \ge S_n) = \sum_{k=S_n}^{n} \binom{n}{k}\gamma^k(1-\gamma)^{n-k}.
\]
- Report both **\(z\)** (for interpretability) and **\(p_{\text{exact}}\)** (for rigor).
- Optionally use **mid-p** to reduce discreteness for borderline cases.

---

### Length Effects & Power

Because \(\text{Std}[S_n] = \Theta(\sqrt{n})\), the **z-score grows like \(\sqrt{n}\)** when the green rate is elevated. Hence, detection strength **increases with length**.

<p align="center">
  <img src="Watermark%20Detection/Plot_Results/Z-score%20Vs%20Text%20Length.png" alt="Z-score growth as a function of generated token length" width="70%"/>
  <br/>
  <em>Figure: z-score increases with token length, improving detection power.</em>
</p>

---

### Sliding-Window Detection (Long Documents)

For long texts, attacks may concentrate edits in segments. Use a **windowed scan**:

1. Choose window size \(w\) (e.g., 100–300 tokens) and stride \(s\) (e.g., 50–100).
2. For each window, compute \(z_w\) (or \(p_{\text{exact}}\)).
3. Aggregate:
   - **Max-z** across windows (powerful but requires multiple-testing control).
   - **Average-z** (stable, slightly less sensitive to local attacks).
   - **Proportion of windows** exceeding threshold.

**Multiple testing control:**
- **FWER** (strict): Bonferroni on \(\alpha / m\) for \(m\) windows.
- **FDR**: Benjamini–Hochberg on window p-values if you prefer sensitivity.

---

### Partial Watermarking / Mixed Texts

If only a portion of the document is watermarked (e.g., copy-paste, edits, quotations), the **max-z** window often exposes the watermarked segment, while the **global z** may be diluted. Consider reporting both **global** and **windowed** decisions.

---

### Robustness to Edits & Paraphrasing

Let \(\rho\) be the **edit rate** (fraction of tokens altered). Roughly,
- The effective sample size is reduced to \((1-\rho) n\).
- The realized green count \(S_n\) decreases, shrinking \(z\).

Empirically, small \(\rho\) (1–10%) has modest impact; paraphrasing has a stronger effect but often leaves **detectable residue** at typical lengths.

<p align="center">
  <img src="Watermark%20Detection/Plot_Results/zscore_empirical_comparison_paraphrased.png" alt="Detection accuracy under paraphrasing and substitutions" width="70%"/>
</p>

---

### Detector Outputs & Recommended Reporting

Return a rich report:
```json
{
  "n": 184,
  "gamma": 0.5,
  "green": 126,
  "z": 6.21,
  "p_value": 2.6e-10,
  "decision": true,                 // z >= z_threshold
  "z_threshold": 4.0,
  "method": "z_one_sided + exact_binom_short",
  "notes": "exact binomial used for n<=80; otherwise z + normal tail"
}
```

---

## 🧰 Repository Layout

- `generation/`: Scripts to generate text with and without watermarking (`generate_watermark.py`, `generate_no_watermark.py`).
- `Watermark Detection/`: Detection pipeline (`detect_watermark.py`), diagnostic plots in `Plot_Results/`, and utility scripts.
- `Paraphrasing_text/`: Text corruption utilities for paraphrasing or token substitutions.
- `evaluation/`: Analysis helpers for clean generations.
- `Results/`: JSONL artifacts from the experiments (z-scores, perplexities, robustness runs).
- `Classifier_Training/`: BERT-based classifier experiments and plots.

---

## 🚀 Getting Started

1. **Install dependencies** (Python 3.8+):
   ```bash
   pip install torch transformers datasets tqdm matplotlib seaborn scikit-learn
   ```

2. **Prepare prompts**: place a prompt list at `data/reddit_prompts.json` (each entry: `{ "prompt_id": int, "text": str }`).

3. **Generate text**:
   - Watermarked: `python generation/generate_watermark.py`
   - Unwatermarked baseline: `python generation/generate_no_watermark.py`

   Outputs are written as JSONL under `data/` with prompt metadata and generated text.

4. **Run detection & plots**:
   ```bash
   python "Watermark Detection/detect_watermark.py"
   ```
   This produces JSONL evaluation files under `Results/` and plots under `Watermark Detection/Plot_Results/` (z-score histograms, ROC/PR curves, perplexity shift, length scaling, confusion matrices).

5. **Robustness checks**: use `Paraphrasing_text/paraphrase_watermarked.py` or `corrupt_watermarked.py` to create paraphrased/substituted samples; then re-run detection.

6. **Black-box classifier**: see `Classifier_Training` for training a BERT detector when the watermark key is unavailable; resulting accuracy plots live in `Classifier_Training/Results/`.

---

## 📊 Experimental Results (GPT-2, γ=0.5)

Detection uses a **z ≥ 4** decision rule unless otherwise noted.

### Clean Generations
- **Default bias (δ≈4.0, 5k prompts):** mean *z* = 27.16 ± 5.61 (watermarked) vs. −0.19 ± 1.46 (unwatermarked); **98.7%** TPR at **0.72%** FPR. Mean Δperplexity (wm − no-wm): **−9.57**.
- **Lower bias (δ≈0.5, 5k prompts):** mean *z* = 11.33 ± 5.48 vs. −0.18 ± 1.41; **84.8%** TPR at **0.46%** FPR. Mean Δperplexity: **+2.13**.
- **Reddit prompts (1k pairs):** mean *z* = 14.44 ± 3.24 vs. −0.10 ± 1.04; **97.9%** TPR with **0%** FPR. Mean Δperplexity: **−33.40**.

### Robustness to Corruptions (1k samples each)
- **Paraphrasing:** mean *z* = 6.96 ± 2.02; **93.9%** TPR at **0.3%** FPR. Δperplexity: **+7.87**.
- **Token substitution (≈1% noise):** mean *z* = 6.48 ± 2.92; **86.3%** TPR at **0.3%** FPR. Δperplexity: **−4.59**.
- **Deletion noise:** mean *z* = 9.11 ± 3.48; **91.0%** TPR at **0.3%** FPR. Δperplexity: **+4.30**.

### Visual Highlights
- **Separation & thresholds:** z-score histograms and ROC/PR curves (`zscore_histogram_grouped.png`, `roc_auc_curve.png`, `precision_recall_curve.png`).
- **Length scaling:** `Z-score Vs Text Length.png` shows growing margin with longer outputs.
- **Quality impact:** perplexity shifts before/after watermarking (`perplexity_shift_gamma=0.5.png`, `ppl_shift_by_zscore_bracket.png`).
- **Robustness:** empirical z-score comparisons under paraphrasing and substitutions (`zscore_empirical_comparison_paraphrased.png`, `zscore_empirical_comparison_substituted_1%.png`).
- **Confusion matrix:** thresholded performance on clean data (`confusion_matrix_reddit.png`).

### Black-Box Classifier (BERT)
Classifier experiments (trained on human vs. model text) further boost detection when the key is unknown. See `Classifier_Training/Results/` for accuracy comparisons across γ, bias, and corruption settings.

---

## 📂 Data Format

All experiment artifacts are **JSONL** where each line is a record:

```json
{ "prompt_id": 42, "prompt": "...", "generated_text": "...", "watermarked_z_score": 12.3, "unwatermarked_z_score": -0.1, "watermarked_ppl": 5.4, "no_watermark_ppl": 5.6, "green_tokens": 126, "total_tokens": 184 }
```

Key fields used in detection and plotting:
- `watermarked_z_score` / `unwatermarked_z_score`: detector scores per sample.
- `watermarked_ppl` / `no_watermark_ppl`: perplexity for quality analysis.
- `green_tokens`, `total_tokens`: raw counts used to recompute z.

---

## ✅ License

This project is released under the MIT License.
