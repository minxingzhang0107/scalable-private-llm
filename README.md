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
└── model_checkpoints/                                      # Fine-tuned models (not included)
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




























