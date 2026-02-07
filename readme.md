# Efficient Information Extraction Using LLMs and Knowledge Distillation: A Study on HPV Health Communication

## Overview

This repository contains the implementation and experimental artifacts for our study on efficient information extraction from HPV-related health communication sources using Large Language Models (LLMs) and Knowledge Distillation (KD).

We investigate how distilled encoder-based student models can retain strong performance while significantly improving computational efficiency. Our best-performing student model is **RoBERTa (encoder-only)**, distilled from a larger teacher model and optimized for multi-label HPV health communication classification.

---

## Repository Structure


- **Data** is available under `/data` containing `train val and test splits`
- **Training and testing code** for our best-performing student model (RoBERTa) is available under `/code`

---

## Installation

We recommend Python 3.9+.

Install dependencies:

```bash
pip install torch transformers datasets scikit-learn pandas
```
## Running the Code
Make sure the data and code are arranged in the same directory (need to be moved)

To train: 
```python
python train.py
```

To test:

```python
python eval.py
```

**This is an evolving repository with citations details of the manuscript to be added after publication**