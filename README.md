# Scalable Private Information Management in Large Language Models

A scalable private information management implementation based on k-Nearest Neighbors Language Models (KNN-LM) with novel data splitting and adaptive selection schemes, utility-preserving and privacy-preserving embedding function design, and forgetting capability via database operations for privacy-preserving large language models.

## Overview

This repository contains the implementation of the framework proposed in the paper "Scalable Private Information Management in Large Language Models."

## Repository Structure
```
scalable-private-llm/
├── README.md                                               # Project documentation
├── environment.yml                                         # Conda environment configuration
├── .gitignore                                              # Git ignore rules
├── src/                                                    # Source code
│   ├── generation/                                         # Answer generation
│   ├── evaluation/                                         # Evaluation 
│   ├── scalability/                                        # Large-scale experiments
│   ├── post_removal_accuracy_on_removed_and_retained/      # Forgetting experiments
│   └── plot/                                               # Plotting 
├── scripts/                                                # Bash scripts (mirror src/ structure)
│   ├── generation/
│   ├── evaluation/
│   ├── scalability/
│   └── post_removal_accuracy_on_removed_and_retained/
├── dataset/                                                # Datasets 
│   ├── public/                                             # Public datasets
│   └── private/                                            # Private datasets
├── results/                                                # Experimental results 
│   ├── public/                                             # Generation results on public data
│   ├── private/                                            # Generation results on private data
│   ├── QA_accuracy/                                        # QA accuracy results
│   └── analysis/                                           # Analysis outputs
└── model_checkpoints/                                      # Trained models
    └── [Download from Google Drive - see README]
```
## Data and Model Downloads

Due to file size limitations, the following resources are hosted on Google Drive:

### Model Checkpoints

Download fine-tuned LoRA adapters and place in `model_checkpoints/`:
- [Download Model Checkpoints (Google Drive)](https://drive.google.com/drive/folders/1U-vXOGvVXNSFT81ON_dY6prMRbbR2ci3?usp=share_link)

**After downloading:**

```bash
# Extract to repository root
unzip model_checkpoints.zip
```

### Datasets

Download datasets and place in `dataset/private/`:
- [Download Synthetic Trajectory Dataset (Google Drive)](https://drive.google.com/drive/folders/1U-vXOGvVXNSFT81ON_dY6prMRbbR2ci3?usp=share_link)

**After downloading:**
```bash
# Extract to dataset directory
unzip syn_traj.zip -d dataset/private/
# Should create: dataset/private/syn_traj/
```
## Requirements

We use an Anaconda environment for our experiments. You can install all required packages directly using the provided `environment.yml` file:
```bash
conda env create -f environment.yml
conda activate [environment_name]
```

## Quick Start

All scripts should be run from the **repository root**:
```bash
cd scalable-private-llm
```
### Evaluation

**Two evaluation metrics: Perplexity and QA Accuracy**

---

#### 1. Perplexity Evaluation

Measures how well models predict test answers. Scripts reference trained models in `model_checkpoints/` (download from Google Drive first).

**(a) LM-Only Perplexity**

Evaluates language model perplexity without KNN datastore.

*Models available:*
- LM trained only on public data
- LM trained on public + private data (without privacy)
- LM trained on public + private data (with DP-SGD)

*Run:*
```bash
bash scripts/evaluation/perplexity/eval_lm_only_perplexity.sh
```

See configuration section in script to select model and dataset.

**(b) KNN-LM Perplexity**

Evaluates KNN-LM perplexity with adaptive selection scheme.

*Embedding models available:*
- Pre-trained embeddings (public data only)
- Fine-tuned embeddings with various privacy-preserving methods:
  - Without privacy protection
  - PI (Private Information) perturbation
  - DDPM (Deidentification via DP Masking)
  - Name perturbation (ε = 0.5, 1, 2, 5, 8, 10)
  - DP-SGD

*Run:*
```bash
# Pre-trained embeddings
bash scripts/evaluation/perplexity/eval_knn_lm_perplexity_dynamic_lambda.sh

# Fine-tuned embeddings
bash scripts/evaluation/perplexity/eval_knn_lm_perplexity_embeddingfinetuned_dynamic_lambda.sh
```

See configuration section in script to select embedding model and test dataset.

**Output:** Perplexity results printed to console.

---

#### 2. QA Accuracy Evaluation

Evaluates answer correctness using GPT-4o-mini semantic similarity comparison.

**Script:** `src/evaluation/QA_accuracy/QA_accuracy_eval.ipynb`

**Note:** Original evaluation uses corporate internal API (not publicly accessible). To reproduce, use OpenAI API with your own key. Evaluation methodology can be used directly—only update API configuration.

**Additional packages:**
```bash
pip install openai langchain langchain-community nltk scikit-learn
```

**Requirements:** OpenAI API key ([obtain here](https://platform.openai.com/api-keys))

**Output:** Results saved to `results/QA_accuracy/comparison_results.csv`































