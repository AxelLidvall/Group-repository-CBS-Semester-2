# Financial Sentiment Classification — Project Plan

## 1. Data
- [ ] Download HuggingFace financial news sentiment dataset (human-labeled, positive/negative/neutral)
- [ ] Explore dataset: check class balance, number of instances, text length distribution
- [ ] Write data description section: instances, features, class distribution, sample rows

## 2. Text Preprocessing
- [ ] Lowercasing
- [ ] Tokenization (`word_tokenize`)
- [ ] Stemming (`PorterStemmer`, following MA2 pattern)
- [ ] Remove stopwords and non-alphabetic tokens
- [ ] Document preprocessing choices and justify them in the paper

## 3. Exploratory Data Analysis
- [ ] Word frequency analysis / word clouds
- [ ] Run topic modeling (LDA) on the corpus to describe what the data is about
- [ ] Include topic modeling results in the data description section
- [ ] Apply NER (spaCy) to identify company names and tickers in the corpus

## 4. Models — Sentiment Classification (HuggingFace labeled data)
Train/test split on HuggingFace dataset, evaluate all models on the same test set.

### Baselines
- [ ] TF-IDF with bigrams + Naive Bayes
- [ ] TF-IDF with bigrams + Logistic Regression
- [ ] VADER lexicon-based baseline (no training needed)

### Embedding-based
- [ ] Word2Vec or GloVe embeddings + MLP (neural network)

### Pre-trained / LLM
- [ ] Fine-tuned FinBERT (pre-trained BERT on financial text)
- [ ] Local LLM via NobodyWho (zero-shot prompting) — see notebook from mandatory assignment
- [ ] Frontier LLM — GPT via OpenAI API (zero-shot prompting)

## 5. Evaluation
- [ ] Use same train/test split for all models
- [ ] Compute: Accuracy, F1-score (macro), Precision, Recall per model
- [ ] Build confusion matrix for each model
- [ ] Compile results into one comparison table
- [ ] Discuss: where do simple models fail? Where do LLMs win?

## 6. Paper Sections (max 15 pages)

### Relevant Literature
- [ ] Find at least one recent paper on financial sentiment analysis or NLP on financial text
- [ ] Summarize main points and compare to your project

### Data Description
- [ ] Number of instances, features, class distribution
- [ ] Sample rows from dataset
- [ ] Results from topic modeling / EDA
- [ ] Justify why this is a real (non-synthetic) dataset

### ML Task
- [ ] Define target variable: 3-class sentiment (positive / negative / neutral)
- [ ] Describe all models clearly enough for another researcher to reproduce
- [ ] Describe preprocessing pipeline
- [ ] Describe evaluation metrics and why they were chosen

### Relevance
- [ ] Argue business value: financial sentiment has downstream value for investment decisions, risk monitoring, trading signals
- [ ] Argue why this task requires genuine language understanding (financial jargon, implicit sentiment)
- [ ] Connect to course themes: spectrum from simple BoW to frontier LLM

### Results & Discussion
- [ ] Present comparison table
- [ ] Discuss which models handle financial language better and why
- [ ] Reflect critically on methodological choices (class imbalance, dataset size, lexicon limitations)

### Code Attribution
- [ ] Note any code from external sources, published papers, or AI-generated code

## 7. Oral Exam Prep
- [ ] Prepare 5-min presentation of the project (all group members contribute)
- [ ] Make sure everyone understands ALL the code used
- [ ] Review entire course curriculum — exam is 50/50 project vs. general course questions
- [ ] Prepare for individual questions directed at specific group members
- [ ] Try to speak equally during the oral exam
