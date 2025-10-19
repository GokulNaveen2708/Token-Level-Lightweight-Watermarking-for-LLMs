
# A Lightweight Watermarking Method for Large Language Models

<p align="center">
  <img src="docs/figures/zscore_distributions.png" alt="Z-score distribution comparison between unwatermarked and watermarked text" width="80%"/>
  <br/>
  <em>Representative z-score separation for unwatermarked vs. watermarked generations.</em>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#-license)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Transformers](https://img.shields.io/badge/🤗-Transformers-ff69b4)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-informational)

> **TL;DR**  
> We watermark text **during generation** by **subtly biasing** token selection (keyed green list favored, red list disfavored) using a **secret key** and a **deterministic per-step vocabulary permutation**. Later, a simple **one-sided z-test** detects the hidden signal with high confidence—while keeping the text fluent and natural.

---

## ✨ Overview

Modern LLMs generate vast amounts of content, making **provenance** and **authenticity** critical for safety, moderation, and trust. This repository contains the **official implementation** of our paper, **“A Lightweight Watermarking Method for Large Language Models.”** Our method is designed to be:

- **Lightweight & Production-Ready:** Pure **inference-time** logit adjustment—**no model fine-tuning** or architectural changes.
- **Accurate & Calibratable:** A keyed **z-score detector** provides strong separation at typical lengths; thresholds (e.g., *z* ≥ 4) can be tuned for desired false-positive rates.
- **Quality-Preserving:** With a small bias (e.g., **δ = 0.5**), measured **perplexity shifts are minimal**, keeping outputs natural.
- **Robust in Practice:** Maintains useful signal under **paraphrasing**, **token substitutions**, and **minor edits**; an optional **BERT classifier** supports black-box detection when keys aren’t available.

**How it works (at a glance):**
- At each decoding step, a **secret key** and a **rolling prefix window** seed a PRNG that **permutes the vocabulary**.
- The top **γ fraction** of the permuted vocab becomes the **green list** (favored), and the rest is **red** (disfavored).
- We add a small **δ** logit bonus to green tokens before sampling.  
- For detection, we **recompute** the per-step green lists with the **same key** and run a **one-sided z-test** on how many realized tokens fell into green.

**What’s in this repo:**
- A plug-and-play **Watermarker** wrapper for Hugging Face causal LMs (e.g., GPT-2).
- A keyed **Z-Score Detector** and an optional **BERT-based classifier** for black-box detection.
- Scripts & notebooks for **generation**, **detection**, **robustness** experiments, and **quality** analysis.
- Ready-to-use result figures under `docs/figures/` to showcase performance (z-score distributions, length scaling, confusion matrix, perplexity shifts, robustness, classifier accuracy).

> **Note:** Place the provided images in `docs/figures/` to render visuals throughout the README. You can regenerate or replace them with your exact experimental outputs later.

---

## 🔬 Method (High Level)

This method embeds a **statistical watermark** during generation by **gently nudging** the model toward a secret, per-step **green list** of tokens. The nudging is tiny (logit bonus **δ**) and keyed, so the pattern is **invisible without the key** but **recoverable** by recomputing the same green lists at detection time.

### Core Idea

At each decoding step *t*, we derive a **seed** from the last *k* tokens (the **rolling prefix**) and a **secret key**. Using this seed, we generate a deterministic **pseudorandom ranking** over the vocabulary and choose the top **γ** fraction as the **green list** \(G_t\). We then **add δ** to logits of tokens in \(G_t\) before sampling.

- **Rolling Prefix:** the last *k* generated tokens (including prompt context if needed).
- **Secret Key:** any secret string/bytes/number known to the generator & detector.
- **γ (gamma):** green-list ratio (e.g., 0.5).
- **δ (delta):** logit bonus applied to green tokens (e.g., 0.5).
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
  <img src="docs/figures/zscore_vs_length.png" alt="Z-score growth as a function of generated token length" width="70%"/>
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
  <img src="docs/figures/robustness_edits.png" alt="Detection accuracy under paraphrasing and substitutions" width="70%"/>
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

