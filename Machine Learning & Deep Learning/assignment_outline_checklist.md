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
- [ ] Describe the FASDD_RS dataset (source, size, image format, rgb(?))
- [ ] Report class distribution: smoke (1,335) vs. neitherFireNorSmoke (888)
- [ ] Report train/val/test split (1,112 / 741 / 370) (with percentages)
- [ ] Include sample images from each class
- [ ] Discuss class imbalance and whether any action will be taken

### 4.2 Data Preprocessing

#### 4.2.1 Data Filtering
- [ ] Check for and remove duplicate images (perceptual hashing)
- [ ] Check for corrupt or unreadable TIF files
- [ ] Check RGB anomalies (near-black or near-white images)

#### 4.2.2 Data Normalisation
- [ ] Resize all images from 1000×1000 to a target size (e.g. 224×224 for pre-trained models)
- [ ] Normalise pixel values to [0, 1] or ImageNet mean/std
- [ ] Confirm train/val/test split is maintained

#### 4.2.3 Data Augmentation
- [ ] Apply augmentations to training set: flip, rotate, brightness, crop, Gaussian noise
- [ ] Report original vs. augmented dataset sizes
- [ ] Visualise sample augmented images

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
