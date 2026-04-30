# Custom CNN Model — Design Plan with Justifications (v8)

## Overview

CNN v8 is a 4-block convolutional network built on the same structural foundation as v6/v7 (4 conv blocks + two-layer dense head), but introduces stronger training-side interventions: revised class weights, increased label smoothing, an expanded augmentation policy, and a warmup + cosine decay learning rate schedule. The architecture itself is identical to v6/v7 — the changes are entirely in how the model is trained.

---

## Architecture Decisions

### Conv Blocks (×4) — Overview

Each block follows the pattern:
`Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU → MaxPool(2×2) → Dropout`

| Block | Filters | Output shape | Dropout |
|-------|---------|--------------|---------|
| 1     | 32      | 112×112×32   | 0.25    |
| 2     | 64      | 56×56×64     | 0.25    |
| 3     | 128     | 28×28×128    | 0.25    |
| 4     | 256     | 14×14×256    | 0.25    |

All conv kernels: 3×3, padding='same'.

### Dense Head — Overview

```
GlobalAveragePooling2D
Dense(512) → BatchNorm → ReLU → Dropout(0.4)
Dense(256) → BatchNorm → ReLU → Dropout(0.5)
Dense(6, activation='softmax')
```

---

### Why 4 conv blocks?

A fourth block (32→64→128→256 filters) is added relative to the baseline 3-block model. With 4 blocks the spatial dimensions reduce to 14×14 after MaxPool, giving the network a larger receptive field and the capacity to detect larger-scale features such as full smoke plumes or dust fronts spanning much of the image. The additional 256-filter block significantly increases representational capacity without the overfitting risk of a fully connected expansion, since GlobalAveragePooling2D still compresses the spatial dimensions before the dense head.

### Why double Conv2D per block?

Two conv layers in a row (before pooling) allows the network to learn more complex, compositional features at each spatial scale without immediately discarding spatial resolution through pooling. This pattern comes from VGG (2014), where it was shown that two 3×3 convolutions cover the same receptive field as one 5×5 convolution, but with fewer parameters and an extra non-linearity. More non-linearities = more expressive power.

### Why 3×3 kernels?

3×3 is the standard in modern CNNs (VGG, ResNet, EfficientNet all use it). It is the smallest kernel that can capture spatial context in all 8 directions around a pixel. Larger kernels (5×5, 7×7) capture the same receptive field with more parameters and no meaningful accuracy gain. Smaller (1×1) kernels have no spatial context at all.

### Why padding='same'?

'same' padding preserves the spatial dimensions after each conv layer (output is same width/height as input). This means spatial reduction only happens at the explicit MaxPool steps, giving full control over the architecture. Without it, each conv layer would shrink the feature maps slightly, making it harder to reason about the architecture and potentially losing border information.

### Why filters 32 → 64 → 128 → 256 (doubling)?

This is the standard progression in CNN design. Early layers detect simple low-level features (edges, colours, textures) — 32 filters is sufficient. Deeper layers need to detect more complex, abstract patterns (smoke texture vs. haze gradient vs. cloud shape), which requires more filters. Doubling at each block is a practical rule of thumb that balances expressiveness and parameter count.

### Why BatchNormalization after each Conv2D?

Batch Normalisation (Ioffe & Szegedy, 2015) normalises the activations within each mini-batch, which:
1. Reduces internal covariate shift — each layer receives inputs with stable distribution, so learning is faster
2. Acts as a mild regulariser, reducing the need for aggressive dropout
3. Allows higher learning rates without divergence

### Why ReLU activation?

ReLU (Rectified Linear Unit) is the default activation for hidden layers because:
1. It does not suffer from the vanishing gradient problem that sigmoid/tanh have — gradient is always 1 for positive inputs
2. It is computationally cheap (just a max(0, x))
3. It introduces sparsity — neurons that don't fire output 0, which acts as a form of regularisation

### Why MaxPool(2×2)?

MaxPool with a 2×2 window and stride 2 halves the spatial dimensions, discarding 75% of spatial positions and keeping only the maximum activation in each 2×2 patch. This achieves:
1. Spatial invariance — small translations in input don't change output
2. Dimensionality reduction — reduces computation in subsequent layers
3. Implicit feature selection — maximum activations correspond to strongest feature detections

### Why Dropout(0.25) after each conv block?

0.25 (25%) is a mild rate appropriate for conv layers — conv layers already have some implicit regularisation due to weight sharing, so aggressive dropout would destroy too much spatial information. Higher rates (0.4, 0.5) are reserved for the dense head where the risk of overfitting is higher.

### Why GlobalAveragePooling2D instead of Flatten?

After 4 conv blocks, the feature maps are 14×14×256. Flattening gives 50,176 values — a large number of parameters that would almost certainly overfit. GlobalAveragePooling2D averages each of the 256 feature maps over all spatial positions, reducing 14×14×256 → 256 values. This:
1. Drastically reduces parameter count in the dense head
2. Provides spatial invariance to the exact position of features
3. Is compatible with Grad-CAM — Grad-CAM requires the spatial structure of the last conv layer to be preserved through to the output, which GAP allows

### Why two-layer dense head (Dense(512) → Dense(256))?

A single dense layer going straight from 256 GAP features to 6 classes would be too abrupt a compression. Two layers allow a graduated reduction: 256 → 512 (expansion for richer combination learning) → 256 → 6 (classification). The intermediate 512-neuron layer can learn more complex combinations before the final decision boundary is drawn. Dropout(0.4) on the first layer and Dropout(0.5) on the second apply increasing regularisation as the head narrows.

### Why softmax output with 6 neurons?

6 neurons correspond to the 6 classes (cloud, dust, haze, land, seaside, smoke). Softmax converts the raw logits into a probability distribution that sums to 1, which is the correct formulation for mutually exclusive multi-class classification.

---

## Training Pipeline Decisions

### Training Pipeline — Overview

| Setting       | Value                                              |
|---------------|----------------------------------------------------|
| Loss          | Label-smoothed cross-entropy (smoothing=0.15)      |
| Optimizer     | AdamW, lr=1e-3                                     |
| Batch size    | 32                                                 |
| Max epochs    | 100                                                |
| LR schedule   | 5-epoch warmup + cosine decay                      |
| Normalisation | divide by 255 → [0, 1]                             |
| Training data | original set (3,735 images) + on-the-fly augmentation |
| Val/Test data | never augmented                                    |

**Callbacks:**
- `EarlyStopping(patience=10, restore_best_weights=True, monitor='val_accuracy')`
- `LearningRateScheduler` — warmup + cosine decay
- `ModelCheckpoint` — saves best val_accuracy weights

---

### Why label smoothing at 0.15 (increased from v7's 0.10)?

Label smoothing prevents the model from becoming overconfident by replacing hard one-hot targets with soft targets. Instead of predicting probability 1.0 for the true class, the target becomes `(1 − α)` for the true class and `α/K` for all others. At `α=0.15` with 6 classes, the true class target is 0.875 and each wrong class receives 0.025. This is stronger than v7's 0.10, justified by the visual similarity between haze, smoke, and dust — overconfident predictions on these classes were the main failure mode in v7.

### Why these class weights?

```
cloud×1.0, dust×1.0, haze×2.0, land×1.0, seaside×0.6, smoke×2.0
```

Class weights scale the loss contribution of each sample. Haze and smoke are the hardest classes to distinguish visually — increasing their weight forces the model to pay more attention to mistakes on these classes. Seaside is an easy class (visually distinct blue water) that the model tends to over-predict, so downweighting it discourages over-reliance on this easy signal.

### Why AdamW at lr=1e-3?

AdamW (Loshchilov & Hutter, 2019) is a corrected version of Adam that decouples weight decay from the gradient update, applying it directly to the weights after the update. This is mathematically correct and leads to better generalisation. lr=1e-3 is the standard default, with the LR schedule handling decay from there.

### Why cosine decay with 5-epoch warmup?

The LR schedule linearly ramps from 0 to `lr=1e-3` over 5 epochs (warmup), then follows a cosine curve down to near-zero by epoch 100. The warmup prevents large gradient updates before the model has stabilised — this solved the exploding validation loss seen in early epochs of v5/v6. Cosine decay then allows smooth, gradual convergence in later epochs without the abrupt drops of `ReduceLROnPlateau`.

### Why EarlyStopping patience=10 (vs 7 in the baseline)?

The cosine decay schedule means val_accuracy can plateau for several epochs mid-decay before improving again as the LR drops into a better region. Patience=7 would terminate training prematurely during these plateaus — patience=10 gives the schedule enough runway to demonstrate whether a lower LR produces gains.

### Why expanded augmentation in `augment_batch_v8`?

v8 adds four new transforms on top of the standard flip/rotate/brightness/noise:
- **Random contrast** `[0.8, 1.2]` — simulates variation in sensor contrast
- **Random saturation** `[0.7, 1.3]` — simulates colour variation in atmospheric imagery
- **Random hue** `±0.05` — small hue shifts for colour calibration differences between captures
- **Noise std reduced** `10/255 → 8/255` — slightly less noise since contrast/saturation already add variation

The extra colour transforms are motivated by the fact that atmospheric classes (haze, smoke, dust) differ partly in colour tone. The model needs to be robust to sensor and lighting variation rather than memorising specific colour signatures.

### Why batch size 32?

Batch size 32 fits comfortably in memory for 224×224 images, provides enough gradient noise to escape local minima, and is large enough for BatchNorm statistics to be meaningful (BatchNorm breaks down at batch size < ~8).

### Why max 100 epochs?

100 epochs is a generous upper bound suited to the cosine decay schedule, which is designed to run for the full duration. EarlyStopping will terminate training earlier if val_accuracy stops improving for 10 consecutive epochs.

---

## Evaluation Decisions

### Why Precision, Recall, and F1?

Accuracy alone is insufficient. For this dataset the classes are roughly balanced, but reporting all three metrics is best practice and expected in academic work:
- **Precision**: of all images predicted as class X, how many actually are X (measures false positives)
- **Recall**: of all actual class X images, how many did the model find (measures false negatives)
- **F1**: harmonic mean of precision and recall — a single balanced score

Macro-average treats all classes equally regardless of size, which is appropriate for a balanced dataset.

### Why a confusion matrix?

The confusion matrix shows exactly which classes are being confused with which others. For satellite imagery this is particularly informative — we expect haze/smoke/dust to be frequently confused because they are visually similar. The confusion matrix confirms or challenges this hypothesis and guides further analysis.

---

## Grad-CAM Decisions

### Why Grad-CAM?

Grad-CAM (Gradient-weighted Class Activation Mapping, Selvaraju et al. 2017) produces a heatmap that highlights which regions of an image most influenced the model's prediction. For remote sensing classification this is crucial for interpretability — we want to confirm the model is looking at the actual atmospheric phenomenon (smoke plume, dust cloud) and not spurious background features.

### Why the last conv layer?

The last conv layer (block 4) contains the most semantically rich feature maps — it has learned high-level representations like "this region looks like smoke" rather than low-level features like edges. Earlier layers capture low-level features (edges, gradients) that are less informative for understanding what the model focuses on for classification.
