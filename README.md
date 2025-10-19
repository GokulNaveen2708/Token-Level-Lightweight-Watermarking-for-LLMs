
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
