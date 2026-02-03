# Scalable Private Information Management in Large Language Models

A scalable private information management implementation based on k-Nearest Neighbors Language Models (KNN-LM) with novel data splitting and adaptive selection schemes, utility-preserving and privacy-preserving embedding function design, and forgetting capability via database operations for privacy-preserving large language models.

## Overview

This repository contains the implementation of the framework proposed in the paper "Scalable Private Information Management in Large Language Models."

## Repository Structure
```
scalable-private-llm/
├── src/                                                    # Source code
│   ├── generation/                                         # Answer generation
│   ├── evaluation/                                         # Evaluation scripts
│   ├── scalability/                                        # Large-scale experiments
│   └── post_removal_accuracy_on_removed_and_retained/      # Forgetting experiments
├── scripts/                                                # Bash scripts (mirror src/ structure)
│   ├── generation/
│   ├── evaluation/
│   ├── scalability/
│   └── post_removal_accuracy_on_removed_and_retained/
├── dataset/                                                # Datasets 
├── results/                                                # Experimental results
└── model_checkpoints/                                      # Fine-tuned models (not included, needs download from Google Drive as explained next)
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

#### QA Accuracy Evaluation

QA accuracy is evaluated using `src/evaluation/QA_accuracy/QA_accuracy_eval.ipynb`. The notebook uses GPT-4o-mini to compare generated answers against ground truth through semantic similarity analysis.

**Note:** The original evaluation uses a corporate internal API and cannot be used by the public. To reproduce the evaluation results, use the OpenAI API with your own API key. The evaluation methodology and prompts in the notebook can be used directly—only the API configuration needs to be updated with your OpenAI credentials.

**Additional packages required:**
```bash
pip install openai langchain langchain-community nltk scikit-learn
```

**Requirements:** OpenAI API key ([obtain here](https://platform.openai.com/api-keys)).

Results are saved to `results/QA_accuracy/comparison_results.csv`.

### Perplexity Evaluation

Evaluates pretrained vs. fine-tuned model perplexity on public/private datasets.

**Models available:**
- LM trained only on public data
- LM trained on public + private data (without privacy)
- LM trained on public + private data (with DP-SGD)

**Run:**
```bash
bash scripts/evaluation/perplexity/eval_lm_only_perplexity.sh
```

See the commented configuration section at the beginning of the bash file to select which model and dataset to use.

**Note:** Scripts reference trained models in `model_checkpoints/`. Download them from Google Drive first. To use your own trained models, replace the model path in the bash file.































