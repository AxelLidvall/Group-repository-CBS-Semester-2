# USTC_SmokeRS Classification - Project Checklist

## 1. Abstract
- [ ] Write abstract summarising problem, models used, key results, and main finding

---

## 2. Introduction
- [ ] Motivate the problem (remote sensing, environmental monitoring use case)
- [ ] State what models will be tested
- [ ] Briefly outline the paper structure

---

## 3. Related Work
- [ ] Find and summarise 3–5 relevant papers on remote sensing image classification
- [ ] Reference CNN-based approaches and transfer learning methods

---

## 4. Methodology

### 4.1 Dataset Description
- [x] Describe the USTC_SmokeRS dataset (source, 6,225 images, 6 classes, 256×256 px)
- [x] Report class distribution across all 6 classes
- [x] Include sample images from each class
- [x] Confirm image dimensions and channel count (RGB, uint8)

### 4.2 Data Preprocessing

#### 4.2.1 Resizing
- [x] Resize all images from 256×256 to 224×224 (lazy loader defined)

#### 4.2.2 Train / Val / Test Split
- [x] Build a unified dataset pool (DataFrame with index + label)
- [x] Perform a fresh stratified 60/20/20 split
- [x] Confirm class ratio is preserved in all three splits
- [x] Report final split sizes (train / val / test) per class

### 4.3 Exploratory Data Analysis (training set only)
- [x] Pixel intensity distributions per channel per class
- [x] Mean brightness and std per class
- [x] Per-image statistics summary table

### 4.4 Data Preprocessing (continued)

#### 4.4.1 Data Filtering
- [x] Check for and remove duplicate images (perceptual hashing)
- [x] Check for corrupt or unreadable images
- [x] Check RGB anomalies: remove near-black (avg < 5) or near-white (avg > 250) images

#### 4.4.2 Data Normalisation
- [x] Define [0, 1] normalisation function (divide by 255) — for Custom CNN and SVM
- [x] Define ImageNet normalisation function — for Transfer Learning only
- [x] Ensure val and test sets use the same normalisation as training for each model

#### 4.4.3 Data Augmentation (training split only)
- [x] Expand training set at 5× with: flip, rotate, brightness, crop, Gaussian noise
- [x] Report dataset sizes before and after augmentation
- [x] Visualise sample augmented images per transform type

#### 4.4.4 Data Pipeline
- [x] Load val and test as numpy arrays with both normalisations ([0,1] and ImageNet)
- [x] Load non-augmented training set with both normalisations (for SVM and Transfer Learning)
- [x] Define `load_aug_split()` for on-demand augmented training loading (for CNN)
- [x] Print pipeline summary confirming all array shapes

### 4.5 Modelling Framework

#### 4.5.1 Custom CNN (from scratch)
- [ ] Design CNN architecture (conv layers, pooling, batch norm, dropout, 6-class dense head)
- [ ] Train on training set, validate on val set
- [ ] Visualise architecture diagram

#### 4.5.2 Pre-trained CNN (Transfer Learning)
- [ ] Select pre-trained backbone (ResNet50 / InceptionV3 / EfficientNet)
- [ ] Freeze base layers, train new 6-class classification head (1 epoch to stabilise)
- [ ] Fine-tune all layers with lower learning rate

#### 4.5.3 SVM (Baseline)
- [ ] Extract features (flattened pixels or CNN features)
- [ ] Apply PCA to reduce dimensionality
- [ ] Perform grid search with 5-fold cross-validation (multi-class one-vs-rest)

### 4.6 Evaluation Metrics
- [ ] Define Precision, Recall, F1-score formulas
- [ ] Justify use of macro-average F1
- [ ] State that test set F1 is the primary evaluation metric

---

## 5. Results

### 5.1 Model Performance
- [ ] Report per-class and macro-average Precision / Recall / F1 for all models
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
- [ ] Show confusion matrix for the best model
- [ ] Show misclassified images and identify patterns

### 6.3 CNN Layer Analysis (Grad-CAM)
- [ ] Apply Grad-CAM to the custom CNN
- [ ] Visualise what regions the model attends to per class
- [ ] Discuss whether the model is looking at relevant features

---

## 7. Conclusion & Future Work
- [ ] Summarise best model and its performance
- [ ] Discuss limitations of the dataset and models
- [ ] Suggest future work

---

## 8. References
- [ ] Compile all citations in consistent format

---

## Appendix
- [ ] A: Data exploration plots (class distribution, sample images)
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
