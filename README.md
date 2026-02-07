# Scalable Private Information Management in Large Language Models

A scalable private information management implementation based on k-Nearest Neighbors Language Models (kNN-LM) with novel data splitting and adaptive selection schemes, utility-preserving and privacy-preserving embedding function design, and forgetting capability via database operations for privacy-preserving large language models.

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

## Getting Started

All scripts should be run from the **repository root**:
```bash
cd scalable-private-llm
```
We provide detailed instructions on Evaluation, Generation, and Training tasks in the following sections. 

### Evaluation

**Two evaluation metrics: Perplexity and QA Accuracy**

---

#### 1. Perplexity Evaluation

**(a) LM-Only Perplexity**

*Models available:*
- LLM trained only on public data
- LLM trained on public + private data (without privacy protection)
- LLM trained on public + private data (with DP-SGD)

*Run:*
```bash
bash scripts/evaluation/perplexity/eval_lm_only_perplexity.sh
```

See configuration section in script to select model and dataset.

**(b) kNN-LM Perplexity**

Evaluates kNN-LM perplexity with adaptive selection scheme.

*Embedding models available:*
- Embedding function trained on public data only
- Embedding functions trained on public data and private data with various privacy-preserving methods:
  - Without privacy protection
  - DP-SGD
  - DDPM (Deidentification via DP Masking)
  - Name perturbation (ε = 0.5, 1, 2, 5, 8, 10)
  - PI (Private Information) perturbation

*Run:*
```bash
# Public data-only embedding
bash scripts/evaluation/perplexity/eval_knn_lm_perplexity_dynamic_lambda.sh

# Public and private data embedding
bash scripts/evaluation/perplexity/eval_knn_lm_perplexity_embeddingfinetuned_dynamic_lambda.sh
```

See configuration section in script to select embedding model and test dataset.

---

#### 2. QA Accuracy Evaluation

**Script:** `src/evaluation/QA_accuracy/QA_accuracy_eval.ipynb`

**Note:** Our evaluation uses corporate internal API (not publicly accessible). To reproduce, use OpenAI API with your own key. Evaluation methodology can be used directly -- only update API configuration.

**Additional packages:**
```bash
pip install openai langchain langchain-community nltk scikit-learn
```

**Requirements:** OpenAI API key ([obtain here](https://platform.openai.com/api-keys))

**Output:** Results saved to `results/QA_accuracy/comparison_results.csv`

---

### Generation

**Note:** Scripts reference trained models in `model_checkpoints/`. Download them from Google Drive first. To use your own trained models, replace the model path in the bash file.

---

#### 1. LLM-Only Generation

**(a) LLM Trained Only on Public Data**

*Run:*
```bash
bash scripts/generation/generation_lm_only_qa.sh
```

See configuration section to select test dataset (public or private).

**(b) LLM Trained on Public and Private Data Without Privacy Protection & LLM Trained on Public and Private Data with DP-SGD**

*Models available:*
- LLM trained on public + private data (without privacy protection)
- LLM trained on public + private data (with DP-SGD)

*Run:*
```bash
bash scripts/generation/generation_finetuned_lm_only_qa.sh
```

See configuration section to select model and test dataset.

---

#### 2. KNN-LM Generation

**(a) Fixed Lambda (The Original KNN-LM Paper)**

Uses fixed interpolation weight between LLM and KNN predictions.

*Run:*
```bash
bash scripts/generation/generation_knn_lm_qa_fixed_lambda.sh
```

See configuration section to select test dataset and lambda value (0.25, 0.5, 0.75).

**Important:** You must also edit the Python script (lines 803-805) to match your configuration. See comments in bash file.

**(b) Adaptive Selection Scheme (Embedding Function Trained Only on Public Dataset)**

*Run:*
```bash
bash scripts/generation/generation_knn_lm_qa_dynamic_lambda.sh
```

See configuration section to select test dataset and distance threshold (0.1-0.8).

**(c) Adaptive Selection Scheme (Embedding Function Trained on Public and Private Datasets with Different Privacy Protection Settings)**

*Embedding models available:*
- On public and private dataset without privacy protection 
- On public and private dataset with DP-SGD
- On public and private dataset with DDPM (Deidentification via DP Masking)
- On public and private dataset with name perturbation (ε = 0.5, 1, 2, 5, 8, 10)
- On public and private dataset with PI (Private Information) perturbation

*Run:*
```bash
bash scripts/generation/generation_knn_lm_qa_embeddingfinetuned_dynamic_lambda.sh
```

See configuration section to select embedding model, test dataset, and distance threshold.

---

**Output:** All generated answers are saved as JSON files in `results/` directory with paths specified in each script's configuration section.

---

### Training

Train LoRA adapters on private datasets with different privacy-preserving methods.

**Note:** Fine-tuned models are saved to `model_checkpoints/`. Download pre-trained adapters from Google Drive or train your own using the scripts below.

---

#### 1. Standard LoRA Fine-tuning

Trains LoRA adapters with privacy-preserving data preprocessing methods.

**Privacy-preserving methods available:**
- On public and private dataset without privacy protection
- On public and private dataset with PI (Private Information) perturbation
- On public and private dataset with name perturbation (ε = 0.5, 1, 2, 5, 8, 10)
- On public and private dataset with DDPM (Deidentification via DP Masking)

*Run:*
```bash
bash scripts/lora_finetune/lora_finetune.sh
```

See configuration section to select dataset with desired privacy-preserving method.

**Note:** This applies privacy protection to the data before training. For privacy protection during training, use DP-SGD (below).

---

#### 2. DP-SGD LoRA Fine-tuning

Trains LoRA adapters with entity-level differential privacy using DP-SGD.

**Privacy guarantee:** User-level DP (adding/removing any single user's data changes the model by at most ε with probability 1-δ)

*Run:*
```bash
bash scripts/lora_finetune/entity_level_DPSGD_lora_efficient.sh
```

**Output:** LoRA adapter saved to `model_checkpoints/user_dp_lora_mistral_[timestamp]/`

---

**Note:** All LoRA fine-tuning scripts save adapters in PEFT format compatible with Hugging Face's `PeftModel.from_pretrained()`.































