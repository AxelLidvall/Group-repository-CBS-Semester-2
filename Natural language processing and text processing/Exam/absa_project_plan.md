# Entity-Aware Financial Sentiment — Project Plan (SEntFiN)

**Task**: Aspect-based sentiment classification of financial news headlines — classify sentiment (positive / negative / neutral) towards a specific named entity within each headline.
**Dataset**: SEntFiN 1.0 (Sinha et al., 2023) — 10,753 headlines, 14,404 entity-sentiment annotations.
**Reference paper**: `2305.12257v1.pdf`

---

## 1. Data

- [ ] Verify SEntFiN 1.0 is loadable (HuggingFace or direct download from paper authors)
- [ ] Load dataset and inspect structure: headline, entity, sentiment label
- [ ] Explore class distribution — paper reports 35.23% positive / 26.48% negative / 38.29% neutral (much more balanced than financial_phrasebank)
- [ ] Explore multi-entity headlines — 2,847 headlines have more than one entity, producing multiple training instances each
- [ ] Write data description: instances, entity count, class distribution, sample rows

---

## 2. Preprocessing

- [ ] Apply Target/Other entity substitution: replace the focal entity with the token `Target`, all other entities in the same headline with `Other` (see Figure 1 in paper)
- [ ] This produces one instance per entity per headline — 14,404 instances total from 10,753 headlines
- [ ] Lowercasing, tokenization (`word_tokenize`), remove punctuation
- [ ] Note: the paper does NOT use stop-word removal for lexicon-based models — follow this choice and document it
- [ ] 80/20 train/test split (~11,200 train / ~2,800 test), same split used for all models

---

## 3. Exploratory Data Analysis

- [ ] Word frequency analysis on Target-substituted headlines
- [ ] Compare headline length distribution for single-entity vs multi-entity headlines
- [ ] Visualize class distribution per entity type (company vs sector vs commodity)
- [ ] Apply NER (spaCy) to explore what kinds of entities appear most frequently

---

## 4. Models — Entity-Aware Sentiment Classification

All models receive the Target/Other substituted headline as input. Evaluated on the same test set.

### Baselines (lexicon / no training)
- [ ] VADER on Target/Other text — replicates Table 4 in the paper (non-entity-aware baseline for comparison)
- [ ] TF-IDF with unigrams + bigrams + trigrams + Naive Bayes — analogous to UBT+GBM in paper
- [ ] TF-IDF with unigrams + bigrams + trigrams + Logistic Regression — analogous to UBT+SVM in paper

### Embedding-based
- [ ] Word2Vec (gensim) averaged vectors + Logistic Regression — analogous to GloVe+LSTM family in paper

### Pre-trained / LLM
- [ ] FinBERT zero-shot on Target/Other text (outside course scope, flagged)
- [ ] Local LLM via NobodyWho — zero-shot prompt includes entity name: `"What is the sentiment towards [ENTITY] in this headline: positive, negative, or neutral?"`
- [ ] Frontier LLM — GPT via OpenAI API, same entity-aware prompt

---

## 5. Evaluation

- [ ] Compute per-class accuracy and F1-score for each model (matches paper's Table 5 format)
- [ ] Compute macro F1 as primary metric
- [ ] Build confusion matrix for each model
- [ ] Compile all results into one comparison table
- [ ] Compare against paper's reported numbers where methodology overlaps
- [ ] Discuss: how much does entity substitution help vs treating the headline as a whole?

---

## 6. Paper Sections (max 15 pages)

### Relevant Literature
- [ ] Summarize SEntFiN 1.0 paper as the primary reference — dataset, methodology, results
- [ ] Note how our model set maps to theirs (NB/LR instead of SVM/GBM, Word2Vec instead of GloVe, adding LLM zero-shot which they do not do)

### Data Description
- [ ] Dataset origin, annotation process, inter-annotator agreement (from paper: 98.26% neg/pos, 96.85% neg/neu, 80.36% pos/neu)
- [ ] Class distribution, entity count, headline length stats
- [ ] Explain what makes this harder than sentence-level: conflicting sentiments for different entities in the same headline

### ML Task
- [ ] Define target: 3-class entity-level sentiment (positive / negative / neutral) per entity per headline
- [ ] Describe Target/Other substitution and why it is necessary
- [ ] Describe all models and preprocessing pipeline
- [ ] Justify macro F1 as primary metric

### Relevance
- [ ] Argue why entity-aware analysis is more actionable than sentence-level for investors (entity-specific signals)
- [ ] Connect to course themes: spectrum from lexicon/BoW to frontier LLM
- [ ] Note that sentence-level baselines (VADER, HuggingFace) perform poorly on this task (Table 4 in paper) — motivates the entity-aware framing

### Results & Discussion
- [ ] Present comparison table
- [ ] Discuss performance on neutral class — hardest class (paper: 80.36% inter-annotator agreement confirms inherent ambiguity)
- [ ] Discuss whether LLMs close the gap with entity-aware prompting

### Code Attribution
- [ ] Note any code from paper, external sources, or AI-generated code

---

## 7. Oral Exam Prep

- [ ] Prepare 5-min presentation (all group members contribute)
- [ ] Make sure everyone understands ALL the code used
- [ ] Review entire course curriculum — exam is 50/50 project vs. general course questions
- [ ] Prepare for individual questions directed at specific group members
- [ ] Try to speak equally during the oral exam
