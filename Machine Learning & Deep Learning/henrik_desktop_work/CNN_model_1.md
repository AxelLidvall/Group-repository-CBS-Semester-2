# Custom CNN Model — Design Plan with Justifications

## Overview

A CNN built from scratch in TensorFlow/Keras for 6-class satellite image classification on USTC_SmokeRS. Input images are 224×224×3, normalised to [0, 1].

The goal is a model complex enough to learn meaningful visual features from satellite imagery, but not so large that it overfits on ~18k training images or takes days to train.

---

## Architecture Decisions

### Conv Blocks (×3) — Overview

Each block follows the pattern:
`Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU → MaxPool(2×2) → Dropout`

| Block | Filters | Output shape | Dropout |
|-------|---------|--------------|---------|
| 1     | 32      | 112×112×32   | 0.25    |
| 2     | 64      | 56×56×64     | 0.25    |
| 3     | 128     | 28×28×128    | 0.25    |

All conv kernels: 3×3, padding='same'.

### Dense Head — Overview

```
GlobalAveragePooling2D
Dense(256) → BatchNorm → ReLU → Dropout(0.5)
Dense(6, activation='softmax')
```

---

### Why 3 conv blocks?

Three blocks strike the balance between underfitting and overfitting for this dataset size. Each block reduces spatial dimensions by half (via MaxPool), so after 3 blocks a 224×224 input becomes 28×28 — still spatially rich enough to capture meaningful features. A 4th block would reduce it to 14×14 and add parameters that the dataset likely cannot support without overfitting. Two blocks would be too shallow to distinguish visually similar classes like haze vs. smoke vs. dust, which require hierarchical feature extraction.

### Why double Conv2D per block?

Two conv layers in a row (before pooling) allows the network to learn more complex, compositional features at each spatial scale without immediately discarding spatial resolution through pooling. This pattern comes from VGG (2014), where it was shown that two 3×3 convolutions cover the same receptive field as one 5×5 convolution, but with fewer parameters and an extra non-linearity. More non-linearities = more expressive power.

### Why 3×3 kernels?

3×3 is the standard in modern CNNs (VGG, ResNet, EfficientNet all use it). It is the smallest kernel that can capture spatial context in all 8 directions around a pixel. Larger kernels (5×5, 7×7) capture the same receptive field with more parameters and no meaningful accuracy gain. Smaller (1×1) kernels have no spatial context at all.

### Why padding='same'?

'same' padding preserves the spatial dimensions after each conv layer (output is same width/height as input). This means spatial reduction only happens at the explicit MaxPool steps, giving us full control over the architecture. Without it, each conv layer would shrink the feature maps slightly, making it harder to reason about the architecture and potentially losing border information.

### Why filters 32 → 64 → 128 (doubling)?

This is the standard progression in CNN design. Early layers detect simple low-level features (edges, colours, textures) — 32 filters is sufficient. Deeper layers need to detect more complex, abstract patterns (smoke texture vs. haze gradient vs. cloud shape), which requires more filters. Doubling at each block is a practical rule of thumb that balances expressiveness and parameter count. Starting at 32 is appropriate for a medium-sized dataset; starting at 64 would double parameter count for no benefit early in the network.

### Why BatchNormalization after each Conv2D?

Batch Normalisation (Ioffe & Szegedy, 2015) normalises the activations within each mini-batch, which:
1. Reduces internal covariate shift — each layer receives inputs with stable distribution, so learning is faster
2. Acts as a mild regulariser, reducing the need for aggressive dropout
3. Allows higher learning rates without divergence

It is placed **before** the activation (ReLU) following the original paper's recommendation, though in practice both orderings work.

### Why ReLU activation?

ReLU (Rectified Linear Unit) is the default activation for hidden layers because:
1. It does not suffer from the vanishing gradient problem that sigmoid/tanh have — gradient is always 1 for positive inputs
2. It is computationally cheap (just a max(0, x))
3. It introduces sparsity — neurons that don't fire output 0, which acts as a form of regularisation

Alternatives like LeakyReLU or ELU exist but ReLU is the safe default and well-justified for image classification.

### Why MaxPool(2×2)?

MaxPool with a 2×2 window and stride 2 halves the spatial dimensions, discarding 75% of spatial positions and keeping only the maximum activation in each 2×2 patch. This achieves:
1. Spatial invariance — small translations in input don't change output
2. Dimensionality reduction — reduces computation in subsequent layers
3. Implicit feature selection — maximum activations correspond to strongest feature detections

AveragePool is an alternative but MaxPool is better for detecting whether a feature is present (which is what classification needs), rather than how strongly it is present on average.

### Why Dropout(0.25) after each conv block?

Dropout randomly sets a fraction of neurons to zero during training, forcing the network to not rely on any single feature. 0.25 (25%) is a mild rate appropriate for conv layers — conv layers already have some implicit regularisation due to weight sharing, so aggressive dropout would destroy too much spatial information. Higher rates (0.5) are reserved for the dense head where the risk of overfitting is higher.

### Why GlobalAveragePooling2D instead of Flatten?

After the 3 conv blocks, the feature maps are 28×28×128. Flattening this gives 28×28×128 = 100,352 values fed into the dense layer — a massive number of parameters that would almost certainly overfit. GlobalAveragePooling2D averages each of the 128 feature maps over all spatial positions, reducing 28×28×128 → 128 values. This:
1. Drastically reduces parameter count in the dense head
2. Provides spatial invariance to the exact position of features
3. Is compatible with Grad-CAM — Grad-CAM requires the spatial structure of the last conv layer to be preserved through to the output, which GAP allows (Flatten destroys the spatial correspondence)

### Why Dense(256)?

256 neurons provides a moderately sized fully connected layer that can learn complex combinations of the 128 features from GAP before the final classification. It is large enough to be expressive but small enough that Dropout(0.5) can regularise it effectively. Common choices are 128, 256, 512 — 256 is a sensible middle ground for a 6-class problem.

### Why Dropout(0.5) in the dense head?

The dense layer is the highest-risk overfitting point — unlike conv layers, dense layers have no weight sharing or spatial structure, so every neuron independently learns combinations of all 256 inputs. 0.5 (50%) dropout is the standard rate for dense layers, originally proposed by Hinton et al. (2012) in the dropout paper, and widely validated in practice.

### Why softmax output with 6 neurons?

6 neurons correspond to the 6 classes (cloud, dust, haze, land, seaside, smoke). Softmax converts the raw logits into a probability distribution that sums to 1, which is the correct formulation for mutually exclusive multi-class classification. The predicted class is the argmax of these 6 probabilities.

---

## Training Pipeline Decisions

### Training Pipeline — Overview

| Setting       | Value                                |
|---------------|--------------------------------------|
| Loss          | sparse_categorical_crossentropy      |
| Optimizer     | AdamW, lr=1e-3                       |
| Batch size    | 32                                   |
| Max epochs    | 50                                   |
| Normalisation | divide by 255 (normalize_standard()) |
| Training data | augmented set (18,675 images)        |
| Val/Test data | never augmented                      |

**Callbacks:**
- `EarlyStopping(patience=7, restore_best_weights=True)`
- `ReduceLROnPlateau(patience=3, factor=0.5)`
- `ModelCheckpoint` — saves best val_accuracy weights

---

### Why sparse_categorical_crossentropy loss?

Cross-entropy is the standard loss for classification — it measures the distance between the predicted probability distribution and the true distribution (a one-hot vector). "Sparse" means labels are provided as integers (0–5) rather than one-hot encoded vectors, which is simpler and saves memory. If we had used one-hot encoded labels, we would use categorical_crossentropy — the math is identical.

### Why AdamW optimiser at lr=1e-3?

AdamW (Loshchilov & Hutter, 2019) is a corrected version of Adam. Adam (Kingma & Ba 2015) combines:
- Momentum: uses a moving average of past gradients to smooth updates
- RMSProp: adapts the learning rate per-parameter based on recent gradient magnitudes

AdamW fixes a flaw in how Adam handles weight decay (L2 regularisation). In Adam, weight decay is added into the gradient before adaptive scaling, meaning the decay is applied inconsistently across parameters. AdamW decouples weight decay from the gradient update — it is applied directly to the weights after the update, which is mathematically correct and leads to better generalisation.

lr=1e-3 (0.001) is the standard default learning rate for Adam-based optimisers and works well as a starting point. ReduceLROnPlateau will reduce it automatically if training plateaus.

### Why batch size 32?

Batch size controls how many samples are processed before updating weights. 32 is a well-validated default that:
1. Fits comfortably in GPU memory even for 224×224 images
2. Provides enough gradient noise to escape local minima (unlike very large batches)
3. Is large enough for BatchNorm statistics to be meaningful (BatchNorm breaks down at batch size < ~8)

Larger batches (128, 256) train faster per epoch but often generalise slightly worse. Smaller batches (8, 16) introduce too much noise.

### Why max 50 epochs?

50 epochs is a generous upper bound — we do not expect to actually reach it. EarlyStopping will terminate training when validation loss stops improving. 50 is set high enough that the model has ample time to converge, but the actual number of epochs will be determined by the data, not this cap.

### Why EarlyStopping with patience=7?

EarlyStopping monitors validation loss and stops training when it has not improved for `patience` consecutive epochs. Patience=7 means the model gets 7 chances to recover from a temporary plateau or spike before training is stopped. This:
1. Prevents overfitting — training is stopped when the model starts memorising training data
2. Saves time — no need to train all 50 epochs if convergence happens at epoch 20
3. `restore_best_weights=True` ensures the final model has the weights from the best epoch, not the last

Patience=7 is chosen because ReduceLROnPlateau fires at patience=3, so the model gets roughly 2 learning rate reductions before early stopping kicks in.

### Why ReduceLROnPlateau with patience=3, factor=0.5?

When training plateaus (validation loss doesn't improve for 3 epochs), this halves the learning rate. A smaller learning rate allows finer weight updates that can escape a plateau and find a better local minimum. Factor=0.5 (halving) is the standard choice — aggressive enough to make a difference but not so aggressive that training destabilises. This works together with EarlyStopping: LR reduces at plateau, and if even a lower LR can't improve things for 7 epochs total, training stops.

### Why divide by 255 for normalisation?

Pixel values are originally integers in [0, 255]. Dividing by 255 maps them to floats in [0, 1]. Neural networks train poorly on large input values because:
1. Large inputs cause large activations and large gradients, leading to unstable training
2. Weight initialisation schemes (He, Glorot) assume inputs are roughly in [-1, 1] or [0, 1]

We use simple [0, 1] normalisation (rather than ImageNet mean/std) for the custom CNN because there is no pretrained model involved — the network learns its own feature representations from scratch, and [0, 1] is a clean, interpretable scale. ImageNet mean/std normalisation is only needed when using pretrained weights that were trained with that normalisation.

### Why train on augmented data (18,675 images)?

The original training set has 3,735 images. 5× augmentation expands this to 18,675. Augmentation reduces overfitting by exposing the model to varied versions of the same images (different flips, rotations, brightness levels, noise), effectively simulating a larger and more diverse dataset. The validation and test sets are never augmented because they must represent the real distribution of unseen data — augmenting them would give an unrealistically optimistic evaluation.

---

## Evaluation Decisions

### Why Precision, Recall, and F1?

Accuracy alone is insufficient because it can be misleading when classes are imbalanced. For this dataset the classes are roughly balanced, but reporting all three metrics is best practice and expected in academic work:
- **Precision**: of all images predicted as class X, how many actually are X (measures false positives)
- **Recall**: of all actual class X images, how many did the model find (measures false negatives)
- **F1**: harmonic mean of precision and recall — a single balanced score

Macro-average treats all classes equally regardless of size, which is appropriate for a balanced dataset.

### Why a confusion matrix?

The confusion matrix shows exactly which classes are being confused with which others. For satellite imagery this is particularly informative — we expect haze/smoke/dust to be frequently confused because they are visually similar. The confusion matrix confirms or challenges this hypothesis and guides further analysis.

---

## Grad-CAM Decisions

### Why Grad-CAM?

Grad-CAM (Gradient-weighted Class Activation Mapping, Selvaraju et al. 2017) produces a heatmap that highlights which regions of an image most influenced the model's prediction. For remote sensing classification this is crucial for interpretability — we want to confirm the model is looking at the actual atmospheric phenomenon (smoke plume, dust cloud) and not spurious background features (ocean colour, land texture).

### Why the last conv layer?

The last conv layer (block 3) contains the most semantically rich feature maps — it has learned high-level representations like "this region looks like smoke" rather than low-level features like edges. Earlier layers capture low-level features (edges, gradients) that are less informative for understanding what the model focuses on for classification.

### Why one example per class?

Six images (one per class) gives a concise visual summary that is easy to include in an academic report. It confirms qualitatively that the model's attention is meaningful for each class type.

---

## New Imports Required

Add to imports cell (`b35350e7`):

```python
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import time
import seaborn as sns
```
