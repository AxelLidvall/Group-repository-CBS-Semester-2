# SVM Model — Design Choices & Hyperparameters

## Overview

SVM (Support Vector Machine) is a classical ML baseline. Unlike the CNN and ResNet50, it does not learn features from images — it requires a hand-crafted feature vector as input. The pipeline is: flatten → PCA → scale → SVM with grid-searched hyperparameters.

---

## Step 1: Feature Representation — Flattened Pixels

**Choice:** Flatten each 224×224×3 image into a 150,528-dimensional vector.

**Why flattened pixels (not CNN features):**
The goal is a standalone baseline that doesn't rely on the CNN or ResNet50 we already trained. Using CNN features would make it a "ResNet50 + SVM" hybrid, not a true SVM comparison point. Flattened pixels give SVM its own independent feature space.

**Normalisation:** Use `X_train_norm` (already in [0, 1] range). Consistent with the CNN baseline; keeps pixel values on a common scale before PCA.

---

## Step 2: PCA (Principal Component Analysis)

**Choice:** `n_components=200`, `random_state=42`

**Why PCA at all:**
Raw flattened images are 150,528 features per sample with only 3,735 training images. SVM with RBF kernel scales as O(n²) to O(n³) in samples — but the curse of dimensionality is the bigger problem: distances between points become meaningless in very high dimensions, and the SVM support vectors become unreliable. PCA projects the data into a lower-dimensional space of maximum variance directions.

**Why 200 components specifically:**
- Satellite imagery has high spatial correlation — adjacent pixels are similar — so a small number of components captures most variance
- 200 components typically explains ~60–80% of variance for this type of image data (verified in the notebook before grid search)
- More components → slower SVM training and higher risk of overfitting; fewer → information loss
- 200 is a standard choice in the literature for image SVM tasks (range: 50–300)

**Why not n_components=0.95 (variance-based):**
Variance-based selection can yield 500–1000+ components for this dataset, making the SVM significantly slower with limited accuracy gain.

**Fitting:** PCA is fitted inside the sklearn Pipeline, so it is re-fitted on the training portion of each CV fold — no data leakage.

---

## Step 3: StandardScaler (after PCA)

**Choice:** `StandardScaler()` applied to PCA output.

**Why:**
PCA components have different variances by construction (the first component has the highest variance, the last has the least). SVM with RBF kernel computes distances between points — if components have very different scales, the high-variance components dominate and the SVM effectively ignores low-variance components. StandardScaler normalises each PCA component to zero mean and unit variance, giving all components equal weight.

**Why not scale before PCA:**
The input is already in [0, 1] so all 150,528 pixel features are on the same scale. Scaling per-pixel before PCA would over-normalise and is not necessary here.

**Order in pipeline:** `PCA → StandardScaler → SVM` — scaler is fitted inside CV folds alongside PCA, so no leakage.

---

## Step 4: SVM Hyperparameters (Grid Search)

### Kernel

**Choices searched:** `linear`, `rbf`

| Kernel | How it works | When it works well |
|---|---|---|
| `linear` | Separates classes with a flat hyperplane | When classes are roughly linearly separable |
| `rbf` | Uses Gaussian similarity — circular decision boundaries | Non-linear problems; standard choice for image data |

`rbf` is expected to win for image classification since class boundaries in pixel space are non-linear. `linear` is included as a fast sanity-check baseline.

`poly` kernel was excluded — rarely competitive with `rbf` on image data and adds another hyperparameter (degree).

### C — Regularisation

**Values searched:** `[0.1, 1, 10, 100]`

C controls the trade-off between a wide margin (more misclassifications allowed) and a narrow margin (fewer misclassifications, risk of overfitting).

- `C=0.1` → strong regularisation, large margin, many misclassifications allowed
- `C=100` → weak regularisation, small margin, fits training data harder
- Log-spaced search covers 3 orders of magnitude — standard practice for SVM

### Gamma — RBF kernel width

**Values searched:** `['scale', 'auto']` (only for `rbf`)

Gamma controls how far the influence of a single training point reaches:
- Low gamma → broad influence, smoother, more generalised boundary
- High gamma → tight influence, complex boundary, risk of overfitting

| Value | Formula | Effect |
|---|---|---|
| `'scale'` | 1 / (n_features × X.var()) | Adapts to feature variance — recommended default |
| `'auto'` | 1 / n_features | Simpler, ignores variance |

`'scale'` is generally preferred after sklearn 0.22. Both are included since after PCA + scaling the variance structure changes.

---

## Step 5: Cross-Validation Strategy

**Choice:** `cv=5` (5-fold stratified cross-validation)

**Why 5 folds:**
- With 3,735 training samples, 5-fold gives ~2,988 training / ~747 validation per fold — large enough for stable estimates
- 5-fold is the standard default; 10-fold would give better estimates but 2× longer grid search
- Stratified by default in sklearn's GridSearchCV for classification — each fold preserves class balance

**Scoring metric:** `accuracy`
All 6 classes are balanced (~600 each in training), so accuracy = macro F1 in this case. Accuracy is used for consistency with how the CNN and ResNet50 were monitored.

**n_jobs=-1:** Use all available CPU cores in parallel. SVM is CPU-bound (no GPU support), so this significantly reduces grid search time.

---

## Step 6: Multi-class Strategy

**Choice:** `decision_function_shape='ovr'` (one-vs-rest)

**Why:** With 6 classes, SVM must extend its binary classification to multi-class. OvR trains 6 binary classifiers (each class vs. all others) and assigns the class with the highest decision score. It is the sklearn default and works well for balanced datasets.

The alternative, one-vs-one (OvO), trains C(C-1)/2 = 15 classifiers and is slower with no meaningful accuracy advantage for balanced 6-class problems.

---

## Pipeline Summary

```
X_train_flat  (3735, 150528)
      │
      ▼
PCA(n_components=200)
      │  (3735, 200)
      ▼
StandardScaler()
      │  zero mean, unit variance
      ▼
SVC(kernel=best, C=best, gamma=best, decision_function_shape='ovr')
      │
      ▼
GridSearchCV(cv=5, scoring='accuracy', n_jobs=-1)
```

---

## Expected Output

- Best hyperparameters from grid search
- Best cross-validation accuracy
- Per-class and macro Precision / Recall / F1 on test set
- Confusion matrix
- Total training time (grid search)
- Saved model: `svm_model.pkl`
