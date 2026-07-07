# Apar AI Competition Solution

## Overview

This repository contains the solution developed for the **Apar AI Competition**.

The objective of the competition is to automatically classify multilingual user feedback from the Apar micromobility platform into the correct support department.

The task is a **3-class text classification problem** using user reviews written mainly in:

- Azerbaijani
- Russian
- English

### Target Labels

The model predicts one of the following departments:

```
technical_support
customer_support
other
```

### Evaluation Metric

The competition metric is:

```
Macro F1-Score
```

---

# Solution Approach

Multiple machine learning and deep learning approaches were evaluated.

The final solution uses a multilingual transformer-based model optimized for multilingual customer feedback classification.

---

# Final Model

## XLM-RoBERTa

Model:

```
xlm-roberta-base
```

Framework:

- PyTorch
- HuggingFace Transformers

Architecture:

```
Input Text
     |
     |
Tokenizer
     |
     |
XLM-RoBERTa Encoder
     |
     |
Classification Head
     |
     |
Predicted Department
```

---

# Experiments

## EXP-001 — LinearSVC Baseline

Feature extraction:

- TF-IDF
- Word analyzer
- Unigram features

Model:

```
LinearSVC
```

Result:

```
Macro F1: 0.854415
```

---

## EXP-004 — Character TF-IDF

Changes:

- Character n-grams
- char_wb analyzer
- ngram range (3,5)

Model:

```
LinearSVC
```

Result:

```
Macro F1: 0.886027
```

Character-level features improved robustness against:

- spelling mistakes
- multilingual variations
- noisy user feedback

---

## EXP-007 — Class Weight Balancing

Changes:

```
class_weight="balanced"
```

Model:

```
LinearSVC
```

Result:

```
Macro F1: 0.887025
```

---

## EXP-008 — Multilingual DistilBERT

Model:

```
distilbert-base-multilingual-cased
```

Result:

```
Macro F1: 0.885115
```

---

## EXP-010 — Final Transformer Solution

Model:

```
xlm-roberta-base
```

Training improvements:

- AdamW optimizer
- Linear warmup scheduler
- Early stopping
- Best checkpoint saving
- Tokenizer artifact saving
- Label encoder saving

Final validation score:

```
Macro F1: 0.912248
```

---

# Training Configuration

```yaml
Model:
  xlm-roberta-base

Max Length:
  128

Batch Size:
  16

Epochs:
  8

Learning Rate:
  2e-5

Optimizer:
  AdamW

Warmup Ratio:
  0.1

Early Stopping:
  patience = 2
```

---

# Validation Strategy

Validation method:

```
Stratified Train / Validation Split
```

Split:

```
80% Training
20% Validation
```

Random seed:

```
42
```

Evaluation metric:

```
Macro F1 Score
```

---

# Project Structure

```
apar-ai/

│
├── transformer/
│   │
│   ├── config.py
│   ├── dataset.py
│   ├── tokenizer.py
│   ├── model.py
│   ├── trainer.py
│   ├── evaluate.py
│   ├── inference.py
│   └── train_transformer.py
│
├── artifacts/
│   │
│   ├── best_model.pt
│   ├── tokenizer/
│   └── label_encoder.pkl
│
├── requirements.txt
│
└── README.md
```

Dataset files are intentionally not included because of competition data usage restrictions.

---

# Installation

Create virtual environment:

```bash
python -m venv .venv
```

Activate environment.

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Training

Run:

```bash
python transformer/train_transformer.py
```

Training pipeline:

1. Loads training data
2. Tokenizes text using XLM-RoBERTa tokenizer
3. Fine-tunes transformer model
4. Evaluates validation Macro F1
5. Saves the best checkpoint

Generated artifact:

```
artifacts/best_model.pt
```

---

# Inference

The inference pipeline loads:

- trained model checkpoint
- tokenizer
- label encoder

and generates predictions for unseen test data.

Output format:

```csv
id,label
17769,technical_support
17770,customer_support
17771,other
```

---

# Dependencies

Main libraries used:

- PyTorch
- HuggingFace Transformers
- Scikit-learn
- Pandas
- NumPy
- Joblib

---

# External Resources

This project uses publicly available pretrained models from the HuggingFace ecosystem.

Model:

```
xlm-roberta-base
```

Main technologies:

- HuggingFace Transformers
- PyTorch

---

# Dataset

Competition datasets are not included in this repository.

The dataset is used only for participation in the Apar AI Competition and remains subject to competition rules and restrictions.

---

# Reproducibility

The training pipeline supports reproducible experiments with:

- fixed random seed
- deterministic validation split
- saved tokenizer
- saved label encoder
- saved best model checkpoint

---

# Final Result

Best validation performance:

```
Model:
XLM-RoBERTa Base

Macro F1 Score:
0.912248
```

---

# License

This repository contains code developed for the Apar AI Competition.

Competition datasets and related files are not redistributed.
