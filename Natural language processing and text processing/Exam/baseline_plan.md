# Baseline Models — Detailed Checklist

Three baseline models share a common setup (data, preprocessing, split) and then diverge.
Complete the shared steps once before moving to any individual model.

---

## Shared Setup

### 1. Environment
- [x] Work in: `main_project.ipynb` in the project root (create if it does not exist)
- [x] No venv needed, using system Python with `--break-system-packages`
- [x] Import shared libraries at top of notebook:
  ```python
  import pandas as pd
  import numpy as np
  import nltk
  from nltk.tokenize import word_tokenize
  from nltk.corpus import stopwords
  from nltk.stem import PorterStemmer
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
  import seaborn as sns
  import matplotlib.pyplot as plt
  ```
- [x] Download required NLTK data: `punkt`, `stopwords`, `vader_lexicon`

### 2. Load Dataset
- [x] Load HuggingFace `financial_phrasebank` dataset (or CSV if downloaded manually)
- [x] Inspect: print shape, column names, first 5 rows
- [x] Check class distribution: count positive / negative / neutral labels
- [x] Plot class distribution as a bar chart

### 3. Preprocessing
- [ ] Write `preprocess(text)` function following MA2 pattern:
  ```python
  def preprocess(text):
      text = text.lower()
      tokens = word_tokenize(text)
      cleaned = [ps.stem(t) for t in tokens if t not in stop_words and t.isalpha()]
      return ' '.join(cleaned)
  ```
- [ ] Apply `preprocess` to the full text column
- [ ] Verify output: print 3 original vs. preprocessed examples side by side
- [ ] Encode labels to integers: `LabelEncoder` from sklearn

### 4. Exploratory Data Analysis
- [ ] Plot word frequency distribution (top 20 words per class)
- [ ] Generate word cloud per sentiment class
- [ ] Run LDA topic modeling (`gensim`) on the corpus, print top terms per topic
- [ ] Apply spaCy NER to a sample, note common company names and tickers
- [ ] Write markdown cell summarising EDA findings

### 5. Train/Test Split
- [ ] Split with `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`
- [ ] Print split sizes and verify class balance in both sets
- [ ] **Fix this split for all models** — all models use the same `X_train`, `X_test`, `y_train`, `y_test`

---

## Model 1: TF-IDF + Naive Bayes

### Vectorization
- [ ] Import `TfidfVectorizer` and `MultinomialNB`
- [ ] Fit `TfidfVectorizer(ngram_range=(1, 2))` on `X_train` only
- [ ] Transform both `X_train` and `X_test` (never fit on test)
- [ ] Print vocabulary size

### Training
- [ ] Instantiate `MultinomialNB()`
- [ ] Fit on `(X_train_tfidf, y_train)`

### Evaluation
- [ ] Predict on `X_test_tfidf`
- [ ] Print `classification_report` with target names `['negative', 'neutral', 'positive']`
- [ ] Print `accuracy_score`
- [ ] Plot confusion matrix as seaborn heatmap (reuse MA2 pattern)
- [ ] Save macro F1 to a results dict for the comparison table

### Write-up cell
- [ ] Markdown cell explaining what Naive Bayes is and why it is a sensible baseline
- [ ] Markdown cell explaining what TF-IDF does vs. raw CountVectorizer
- [ ] Markdown cell: how does this answer the classification task?

---

## Model 2: TF-IDF + Logistic Regression

### Vectorization
- [ ] Reuse the **same fitted** `TfidfVectorizer` from Model 1 (do not refit)
- [ ] `X_train_tfidf` and `X_test_tfidf` are already available

### Training
- [ ] Import `LogisticRegression`
- [ ] Instantiate `LogisticRegression(max_iter=1000, random_state=42)`
- [ ] Fit on `(X_train_tfidf, y_train)`

### Evaluation
- [ ] Predict on `X_test_tfidf`
- [ ] Print `classification_report` with target names
- [ ] Print `accuracy_score`
- [ ] Plot confusion matrix as seaborn heatmap
- [ ] Save macro F1 to results dict

### Write-up cell
- [ ] Markdown cell comparing Logistic Regression to Naive Bayes (discriminative vs. generative)
- [ ] Markdown cell: how does this answer the classification task?

---

## Model 3: VADER Lexicon Baseline

> No training needed. VADER runs directly on the raw (unpreprocessed) text.

### Setup
- [ ] Import `SentimentIntensityAnalyzer` from `nltk.sentiment.vader`
- [ ] Instantiate: `sia = SentimentIntensityAnalyzer()`
- [ ] Note in a markdown cell: VADER uses raw text (not stemmed/lowercased) because it relies on capitalisation and punctuation for scoring

### Scoring
- [ ] Write a function to map VADER compound score to a label:
  ```python
  def vader_label(text):
      score = sia.polarity_scores(text)['compound']
      if score >= 0.05:
          return 'positive'
      elif score <= -0.05:
          return 'negative'
      else:
          return 'neutral'
  ```
- [ ] Apply to `X_test` (raw text, not preprocessed)
- [ ] Encode VADER predictions with the same `LabelEncoder` used earlier

### Evaluation
- [ ] Print `classification_report` with target names
- [ ] Print `accuracy_score`
- [ ] Plot confusion matrix as seaborn heatmap
- [ ] Save macro F1 to results dict
- [ ] Note: VADER is evaluated on the test set only (no training, no train set involvement)

### Write-up cell
- [ ] Markdown cell explaining what VADER is and how the compound score works
- [ ] Markdown cell: why is a rule-based lexicon a useful lower bound?
- [ ] Markdown cell: what are VADER's known weaknesses on financial text (jargon, implicit sentiment)?
- [ ] Markdown cell: how does this answer the classification task?

---

## Comparison Table

- [ ] Compile results dict into a `pd.DataFrame` with columns:
  `Model | Accuracy | Macro F1 | Macro Precision | Macro Recall`
- [ ] Print table
- [ ] Write a markdown discussion cell:
  - Which model performs best and why?
  - Where does VADER fail vs. trained models?
  - What does this suggest about financial language understanding?

---

## Checklist Summary

| Step | Status |
|---|---|
| Notebook created | [ ] |
| Dataset loaded and explored | [ ] |
| Preprocessing function written and verified | [ ] |
| EDA complete (word freq, word cloud, LDA, NER) | [ ] |
| Train/test split fixed | [ ] |
| TF-IDF + Naive Bayes trained and evaluated | [ ] |
| TF-IDF + Logistic Regression trained and evaluated | [ ] |
| VADER baseline evaluated | [ ] |
| Comparison table compiled | [ ] |
| Write-up cells complete for all models | [ ] |
