LLM Watermarking
This repository contains a lightweight watermarking framework for Large Language Models (LLMs) using a token-biasing approach called the Left Hash strategy. The aim is to embed detectable patterns in generated text while maintaining fluency and naturalness. The project includes code for watermark generation, detection, evaluation, and robustness testing under text edits such as paraphrasing, substitution, and deletion.

Features
Generate text with and without watermarking using GPT-2.

Detect watermarks using statistical z-score methods and classifier-based detection.

Evaluate text quality with perplexity and grammar metrics.

Test watermark robustness under common edits.

Visualize detection accuracy, perplexity shifts, and token patterns.

Repository Structure
LLM_watermarking/
data/ – Contains prompt datasets and generated samples.

evaluation/
analyse_no_watermark.py – Evaluation script for unwatermarked text.

generation/
generate_no_watermark.py – Generate plain text without watermark.
generate_watermark.py – Generate text with watermark applied.

Paraphrasing_text/
corrupt_watermarked.py – Apply random deletion or substitution to watermarked text.
paraphrase_watermarked.py – Paraphrase watermarked text for robustness testing.

Results/ – Evaluation outputs stored in JSONL format for various test conditions.

Watermark Detection/
Results/ – Plots and figures for detection experiments.
detect_watermark.py – Detect watermarks using z-score statistical analysis.
z-score_Highlight.py – Highlight green tokens in generated text.
classifier_dataset_script.py – Prepare datasets for classifier-based detection.
load_prompts.py – Load general prompt datasets.
load_reddit.py – Load Reddit-specific prompts.
watermarking_architecture.png – Diagram of the watermarking pipeline.

Classifier Training/
BERT_Classifier.py - Trained a BERT Model with watermarked and Human text
predict-with_BERT.py - Detection performance under different thresholds
Classifier_training.py - Trained Random forest model to understand green-red list ratio

Running the Code
To generate watermarked text:
python generation/generate_watermark.py

To generate unwatermarked text:
python generation/generate_no_watermark.py

To detect a watermark in generated text:
python "Watermark Detection"/detect_watermark.py

Plots and visualizations are located in Watermark Detection/Results.