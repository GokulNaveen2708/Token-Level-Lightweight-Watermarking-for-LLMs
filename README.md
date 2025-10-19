
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
