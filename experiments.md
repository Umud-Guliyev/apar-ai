# Apar AI Competition Experiments

---

## EXP-001 — Baseline

**Date:** 2026-07-07

### Objective
Establish the first reproducible baseline for the competition.

### Model
- LinearSVC

### Features
- feedback

### Vectorizer
- TF-IDF
- analyzer = word
- ngram_range = (1,1)

### Validation
- Stratified Train/Validation Split (80/20)

### Result

Macro F1: **0.854415**

### Decision

✅ Accepted as initial baseline.

### Notes

This is the first working pipeline. Future experiments will be compared against this score.

---

## EXP-002 — Feedback + Tag

**Date:** 2026-07-07

### Objective

Check whether concatenating the optional `tag` field with `feedback` improves classification.

### Changes

- Combined `feedback` and `tag`
- Missing tags replaced with empty string

### Result

Macro F1: **0.854415**

### Improvement

0.000000

### Decision

❌ Rejected

### Reason

No measurable improvement over baseline.

---

## EXP-003 — Text Preprocessing Pipeline

**Date:** 2026-07-07

### Objective

Introduce a reusable preprocessing pipeline without changing model behavior.

### Changes

Added `clean_text()`:

- lowercase
- remove duplicated whitespace
- trim spaces

Integrated preprocessing into `TfidfVectorizer(preprocessor=clean_text)`.

### Result

5-Fold Cross Validation

Mean Macro F1: **0.854216**

Std: **0.005341**

### Improvement

Architecture improvement only.

### Decision

✅ Accepted

### Reason

Performance remained stable while code became modular and reusable.

---

## EXP-004 — Character TF-IDF

**Date:** 2026-07-07

### Objective

Evaluate character-level TF-IDF to better handle multilingual text, spelling mistakes and noisy user feedback.

### Changes

Vectorizer configuration:

- analyzer = char_wb
- ngram_range = (3,5)

Model remained:

- LinearSVC

Validation:

- Stratified 5-Fold Cross Validation

### Result

Fold 1: 0.894172

Fold 2: 0.883838

Fold 3: 0.893509

Fold 4: 0.883326

Fold 5: 0.875288

Mean Macro F1: **0.886027**

Std: **0.007068**

### Improvement

+0.031811 Macro F1

### Decision

✅ Accepted

### Reason

Character n-grams significantly improved robustness against spelling variations and multilingual noisy text.

This becomes the new official baseline.

---


## EXP-005 — Logistic Regression


**Date:** 2026-07-07

### Hypothesis

Evaluate whether Logistic Regression performs better than LinearSVC on character-level TF-IDF features.

### Changes

- Model: LogisticRegression(max_iter=3000)
- Character TF-IDF (char_wb, 3-5)
- 5-Fold Stratified Cross Validation

### Result

Fold 1: 0.882760

Fold 2: 0.872957

Fold 3: 0.886500

Fold 4: 0.883186

Fold 5: 0.865910

Mean Macro F1: **0.878263**

Std: **0.007656**

### Improvement

-0.007764

### Decision

❌ Rejected

### Analysis

LinearSVC provides better class separation on sparse character TF-IDF features.
Logistic Regression consistently underperformed across all folds.

## EXP-006 — Word + Character TF-IDF

**Date:** 2026-07-07

### Hypothesis

Combining word-level and character-level TF-IDF features may improve performance by capturing both semantic information (whole words) and robustness to spelling mistakes (character n-grams).

### Changes

Feature pipeline:

- Word TF-IDF
  - analyzer = word
  - ngram_range = (1,2)

- Character TF-IDF
  - analyzer = char_wb
  - ngram_range = (3,5)

- Combined using FeatureUnion

Model:

- LinearSVC

Validation:

- Stratified 5-Fold Cross Validation

### Result

Fold 1: 0.887463

Fold 2: 0.879676

Fold 3: 0.897363

Fold 4: 0.885084

Fold 5: 0.873242

Mean Macro F1: **0.884566**

Std: **0.008057**

### Improvement

-0.001461

### Decision

❌ Rejected

### Analysis

Adding word-level TF-IDF did not improve the model. Character-level TF-IDF alone captures spelling variations and multilingual noise more effectively for this dataset. The additional word features introduced more noise than useful information.

### Conclusion

Character-level TF-IDF remains the best feature representation.

## EXP-007 — Class Weight Balancing

**Date:** 2026-07-07

### Hypothesis

The dataset is moderately imbalanced, with `technical_support` appearing much more frequently than the other classes. Using balanced class weights may improve Macro F1 by giving more importance to minority classes.

### Changes

Model:

- LinearSVC
- class_weight="balanced"

Features:

- Character TF-IDF
- analyzer = char_wb
- ngram_range = (3,5)

Validation:

- Stratified 5-Fold Cross Validation

### Result

Fold 1: 0.895613

Fold 2: 0.884143

Fold 3: 0.894194

Fold 4: 0.885994

Fold 5: 0.875181

Mean Macro F1: **0.887025**

Std: **0.007413**

### Improvement

+0.000998

### Decision

✅ Accepted

### Analysis

Applying balanced class weights slightly improved Macro F1 while maintaining stable cross-validation performance. The improvement is small but consistent enough to keep as the new baseline.

### Conclusion

The new production baseline uses:

- LinearSVC
- class_weight="balanced"
- Character TF-IDF (char_wb)
- ngram_range=(3,5)

---

## EXP-008 — DistilBERT Multilingual Transformer

**Date:** 2026-07-07

### Hypothesis

A multilingual Transformer model pretrained on many languages can better capture the semantic meaning of multilingual user feedback compared to sparse TF-IDF features.

### Model

- DistilBERT Multilingual
- Model: `distilbert-base-multilingual-cased`

### Tokenizer

- DistilBERT Multilingual Tokenizer
- Max Length: 128

### Training

- Optimizer: AdamW
- Learning Rate: 2e-5
- Epochs: 3
- Batch Size: 16
- Train / Validation Split: 80 / 20
- Random Seed: 42

### Validation Result

| Epoch | Validation Macro F1 |
|-------:|--------------------:|
| 1 | 0.870079 |
| 2 | 0.870067 |
| 3 | **0.879706** |

### Best Score

**Macro F1: 0.879706**

### Improvement

Compared to the current TF-IDF + LinearSVC baseline:

0.879706 − 0.887025 = **−0.007319**

### Decision

❌ Rejected

### Analysis

Although the multilingual Transformer successfully learned the task and achieved stable convergence, it did not outperform the existing character-level TF-IDF + LinearSVC baseline.

Possible reasons include:

- Limited training data for fine-tuning a Transformer.
- Only three training epochs were used.
- No learning-rate scheduling or early stopping.
- Default hyperparameters without optimization.
- DistilBERT has fewer parameters than larger multilingual models.

### Conclusion

The current production baseline remains:

- Character TF-IDF (`char_wb`, 3–5)
- LinearSVC
- `class_weight="balanced"`

The Transformer pipeline is now fully implemented and can be further improved through hyperparameter tuning and architectural enhancements in future experiments.

---

## EXP-009 — Transformer Scheduler + Warmup + Early Stopping

**Date:** 2026-07-07

### Hypothesis

Transformer fine-tuning performance can be improved by using a learning rate scheduler with warmup and early stopping.

### Changes

Training improvements:

- AdamW optimizer
- Linear learning rate scheduler
- Warmup steps = 10% of total steps
- Early stopping with patience=2
- Increased maximum epochs from 3 to 5

Model:

- DistilBERT multilingual
- distilbert-base-multilingual-cased

### Validation Result

| Epoch | Macro F1 |
|------:|---------:|
| 1 | 0.825155 |
| 2 | 0.876053 |
| 3 | 0.882393 |
| 4 | 0.884948 |
| 5 | **0.885115** |

### Best Score

**Macro F1: 0.885115**

### Improvement

Compared to EXP-008:

+0.005409

### Decision

✅ Accepted

### Analysis

Learning rate scheduling and warmup improved Transformer fine-tuning stability. The model achieved better convergence and surpassed the previous Transformer baseline.

However, the character-level TF-IDF + LinearSVC model remains slightly stronger.

Current best models:

1. Character TF-IDF + LinearSVC  
   Macro F1: 0.887025

2. DistilBERT multilingual Transformer  
   Macro F1: 0.885115