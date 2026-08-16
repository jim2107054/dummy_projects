# Reproducibility Guide — Nascenia Bengali Medical Dialogue Generation

This guide provides instructions to reproduce the leaderboard submission and verify the parameter-count compliance of the final model.

## Approach Summary

The solution utilizes a **two-stage QLoRA fine-tuning** workflow on the 3B parameter model `hishab/titulm-llama-3.2-3b-v2.0` (which has a continued pre-trained Bangla tokenizer with ~42k Bangla vocabulary tokens).
- **Stage 1 (Broad Mix SFT)**: The model is fine-tuned on a merged broad medical vocabulary mix consisting of the cleaned competition data combined with the normalized external Hugging Face dataset `shetumohanto/doctor_qa_bangla`. This stage builds clinical vocabulary coverage.
- **Stage 2 (Official Anchor SFT)**: The Stage 1 adapter is further fine-tuned at a lower learning rate ($5 \times 10^{-5}$) exclusively on the cleaned official `train.csv` (excluding the validation split) to anchor the generation register, boilerplate/template style, and length distribution back to the target competition distribution.
- **Inference with MBR and Length Calibration**: During generation, the model samples $k=4$ candidates. These candidates are length-calibrated (word limits between 65 and 145) and then reranked using **Minimum Bayes Risk (MBR)** over pairwise ROUGE-L similarity, selecting the most canonical response.

---

## Environment Setup

The notebooks are designed to run in a single-GPU Kaggle Notebook environment (e.g., T4 x1 or A100 x1). The necessary libraries are installed at the beginning of each notebook:

- **Python**: `3.10` or higher
- **Core Libraries**:
  - `unsloth` (for memory-efficient and fast training/inference)
  - `transformers`
  - `trl`
  - `peft`
  - `bitsandbytes`
  - `accelerate`
  - `datasets`
  - `bert-score`
  - `rouge-score`

---

## Step-by-Step Reproduction Instructions

Please execute the notebooks in the following order:

### 1. Data Ingestion & Splitting (`01_data_pipeline.ipynb`)
- **What it does**: Loads `train.csv` and `test.csv`, cleans official boilerplate stub answers, loads and normalizes the external Hugging Face dataset, merges and deduplicates external data, and splits the official clean data into SFT train and validation splits (1,000 rows stratified by length).
- **Outputs**:
  - `/kaggle/working/sft_train.csv` (clean training set)
  - `/kaggle/working/sft_val.csv` (held-out validation set)
  - `/kaggle/working/train_plus_external_clean.csv` (Stage 1 training set)

### 2. Two-Stage QLoRA SFT (`02_train_qlora.ipynb`)
- **What it does**: Loads the base model in 4-bit, applies PEFT adapters, runs Stage 1 training (1 epoch, learning_rate=2e-4), runs Stage 2 training (1 epoch, learning_rate=5e-5), merges the LoRA weights, and validates compliance with the $\leq 3B$ parameter limit.
- **Outputs**:
  - `/kaggle/working/final_model/` (merged FP16 checkpoint)

### 3. Inference and Leaderboard Submission (`03_inference_and_submit.ipynb`)
- **What it does**: Runs model validation predictions over `sft_val.csv` using $k=4$ MBR-reranked generation, validates it against the local metric harness, and generates the final submission over `test.csv`.
- **Outputs**:
  - `/kaggle/working/val_predictions.csv`
  - `/kaggle/working/submission.csv`

### 4. Local Validation (`04_local_validation.ipynb`)
- **What it does**: Evaluates the validation predictions against the composite scoring formula (`0.5*BERTScore_F1 + 0.3*Token_F1 + 0.2*ROUGE-L_F1`) and prints the top 10 worst-scoring validation rows for manual analysis.
- **Outputs**: Detailed metrics printed to console.

### 5. Phase 2 Packaging (`05_phase2_packaging.ipynb`)
- **What it does**: Performs final compliance checks on the merged parameter counts, runs a demo inference, writes out `model_card.md`, and zips the model directory, code modules, and notebooks into a submission archive.
- **Outputs**:
  - `/kaggle/working/phase2_submission.zip`

### 6. Retrieval Add-on (Optional) (`06_retrieval_addon.ipynb`)
- **What it does**: Explores back-translating the Bengali queries to English to perform semantic search over the public HealthCareMagic dataset, and evaluates blending the model outputs with retrieved templates.

---

## Open-Source License

This submission is distributed under the **Apache-2.0 License** in compliance with competition requirements. See the [`LICENSE`](file:///y:/4-1/lab%20slides/al%20mahmud%20sir/hackathon/LICENSE) file at the project root for details.

---

## Limitations and Safety Warning

- **Templated Reference bias**: The model's decoding strategy (length calibration and MBR) is heavily optimized to match the machine-translated reference distributions of Nascenia's Phase 1 evaluation metrics.
- **Not a Clinical Tool**: This model is a competition entry and is trained on translated dialogue datasets. It is not intended for real-world diagnostic use or clinical decision support. Always consult a licensed healthcare professional for medical advice.
