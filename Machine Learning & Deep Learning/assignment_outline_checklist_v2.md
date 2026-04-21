# Smoke Detection - Project Checklist

## 1. Abstract
- [ ] Write abstract summarising problem, models used, key results, and main finding

---

## 2. Introduction
- [ ] Motivate the problem (wildfire detection, remote sensing use case)
- [ ] State what models will be tested
- [ ] Briefly outline the paper structure

---

## 3. Related Work
- [ ] Find and summarise 3–5 relevant papers on smoke/fire detection or similar image classification tasks
- [ ] Reference CNN-based approaches and transfer learning methods

---

## 4. Methodology

### 4.1 Dataset Description

#### 4.1.1 FASDD_RS
- [ ] Describe the FASDD_RS dataset (source, size, image format)
- [ ] Report class distribution: smoke (1,335) vs. neitherFireNorSmoke (888)
- [ ] Include sample images from each class
- [ ] Discuss class imbalance

#### 4.1.2 USTC_SmokeRS
- [ ] Describe the USTC_SmokeRS dataset (source, 6,225 images, 6 classes)
- [ ] Report class distribution across all 6 classes
- [ ] Define binary remapping: smoke → class 1, cloud/dust/haze/land/seaside → class 0
- [ ] Check image dimensions and channel count
- [ ] Visual inspection: confirm images are remote sensing imagery compatible with FASDD_RS
- [ ] Report remapped class counts (smoke vs. background after remapping)

### 4.1.3 Exploratory Data Analysis (per dataset, before combining)
- [ ] Sample image grid per class (FASDD_RS)
- [ ] Pixel intensity distributions per channel (FASDD_RS smoke vs. background)
- [ ] Mean brightness and std per class (FASDD_RS)
- [ ] Per-image statistics summary table (FASDD_RS)
- [ ] Pixel statistics comparison: FASDD_RS vs. USTC_SmokeRS (check domain overlap)

### 4.2 Data Preprocessing

#### 4.2.1 Data Combination
- [ ] Resize all USTC_SmokeRS images from native size to 224×224 (match FASDD_RS target)
- [ ] Resize all FASDD_RS images from 1000×1000 to 224×224
- [ ] Apply binary remapping to USTC_SmokeRS labels
- [ ] Merge both datasets into a single image pool (path/array + label)
- [ ] Report combined class distribution before splitting

#### 4.2.2 Data Filtering
- [ ] Check for and remove duplicate images across both datasets (perceptual hashing)
- [ ] Check for corrupt or unreadable files (FASDD_RS TIFFs + USTC PIL images)
- [ ] Check RGB anomalies: remove near-black (avg < 5) or near-white (avg > 250) images

#### 4.2.3 Train / Val / Test Split
- [ ] Perform a fresh stratified 70/15/15 split on the combined pool
- [ ] Confirm class ratio is preserved in all three splits
- [ ] Report final split sizes (train / val / test) per class

#### 4.2.4 Data Normalisation
- [ ] Normalise pixel values to [0, 1] or ImageNet mean/std
- [ ] Apply ImageNet normalisation (mean/std) for pre-trained model inputs

#### 4.2.5 Data Augmentation (training split only)
- [ ] Phase 1 — oversample minority class only until ratio = 1:1
- [ ] Phase 2 — expand full balanced training set at 5× with: flip, rotate, brightness, crop, Gaussian noise
- [ ] Report dataset sizes at each stage: original → after Phase 1 → after Phase 2
- [ ] Visualise sample augmented images per transform type

### 4.3 Modelling Framework

#### 4.3.1 Custom CNN (from scratch)
- [ ] Design CNN architecture (conv layers, pooling, batch norm, dropout, dense head)
- [ ] Train on training set, validate on val set
- [ ] Visualise architecture diagram

#### 4.3.2 Pre-trained CNN (Transfer Learning)
- [ ] Select pre-trained backbone (e.g. ResNet50, InceptionV3, EfficientNet)
- [ ] Freeze base layers, train new classification head (1 epoch to stabilise)
- [ ] Fine-tune all layers with lower learning rate

#### 4.3.3 SVM (Baseline)
- [ ] Extract features (e.g. flattened pixels or CNN features)
- [ ] Apply PCA to reduce dimensionality
- [ ] Perform grid search with 5-fold cross-validation to tune kernel and hyperparameters

### 4.4 Evaluation Metrics
- [ ] Define Precision, Recall, F1-score formulas
- [ ] Justify use of macro-average F1 (handles class imbalance equally)
- [ ] State that test set F1 is the primary evaluation metric

---

## 5. Results

### 5.1 Model Performance
- [ ] Report Precision, Recall, F1-score per class and macro-average for all models
- [ ] Present results in a comparison table
- [ ] Plot training and validation loss curves for neural models

### 5.2 Complexity & Running Time
- [ ] Record and report training time for each model
- [ ] Discuss trade-off between accuracy and speed

---

## 6. Discussion

### 6.1 Comparison of Models
- [ ] Compare all models on F1 and running time
- [ ] Explain which model performs best and why

### 6.2 Error Analysis
- [ ] Show misclassified images for the best model
- [ ] Identify patterns in errors (e.g. thin smoke, haze, cloud confusion)

### 6.3 CNN Layer Analysis (Grad-CAM)
- [ ] Apply Grad-CAM to the custom CNN
- [ ] Visualise what regions the model attends to for smoke vs. background
- [ ] Discuss whether the model is looking at relevant features

---

## 7. Conclusion & Future Work
- [ ] Summarise best model and its performance
- [ ] Discuss limitations of the dataset and models
- [ ] Suggest future work (e.g. adding fire class, real-time deployment, larger dataset)

---

## 8. References
- [ ] Compile all citations in consistent format

---

## Appendix
- [ ] A: Data exploration plots (class distribution, image size, aspect ratio)
- [ ] B: CNN model architecture diagram
- [ ] C: Pre-trained model details
- [ ] D: Full training/validation accuracy and loss curves
- [ ] E: Confusion matrices for all models
- [ ] F: Additional Grad-CAM visualisations

---

## Code Deliverables
- [ ] Data loading and preprocessing script
- [ ] Custom CNN model definition and training
- [ ] Transfer learning model definition and training
- [ ] SVM pipeline (feature extraction + PCA + grid search)
- [ ] Evaluation script (metrics, confusion matrix, Grad-CAM)
- [ ] Notebook or script to reproduce all results
