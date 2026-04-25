# ResNet50 Transfer Learning — Model Plan

## What is ResNet50?

ResNet50 is a 50-layer convolutional neural network trained on ImageNet (1.2M images, 1,000 classes). "Res" stands for **residual connections** — skip connections that let gradients flow directly past layers, which is what made it possible to train 50+ layers without the gradient vanishing. We get this entire learned feature extractor for free and just retrain the top for our 6 classes.

---

## Training Strategy: Two Phases

Transfer learning works in two phases. Training everything from scratch in one shot would destroy the pretrained weights before the new head has learned anything useful.

```
Phase 1 — Head Only        Phase 2 — Fine-tuning
─────────────────────      ─────────────────────
ResNet50 backbone  FROZEN  ResNet50 last ~30 layers  UNFROZEN
Dense head         TRAIN   Dense head                 TRAIN

High LR (1e-3)             Low LR (1e-5)
~15 epochs                 ~15 epochs
```

---

## Network Blueprint

```
Input
  │  224 × 224 × 3  (ImageNet-normalised RGB)
  ▼
┌─────────────────────────────────────────────────────────┐
│  ResNet50 Backbone  (pretrained on ImageNet)            │
│                                                         │
│  Stage 1:  Conv 7×7, 64 filters, stride 2              │
│            MaxPool 3×3, stride 2                        │
│            → 56 × 56 × 64                              │
│                                                         │
│  Stage 2:  3 × Residual block (64 filters)             │
│            → 56 × 56 × 256                             │
│                                                         │
│  Stage 3:  4 × Residual block (128 filters)            │
│            → 28 × 28 × 512                             │
│                                                         │
│  Stage 4:  6 × Residual block (256 filters)            │
│            → 14 × 14 × 1024                            │
│                                                         │
│  Stage 5:  3 × Residual block (512 filters)            │
│            → 7 × 7 × 2048                              │
│                                                         │
│  [Phase 2: only Stage 5 + last block of Stage 4        │
│   unfrozen — earlier layers stay frozen]               │
└─────────────────────────────────────────────────────────┘
  │  7 × 7 × 2048
  ▼
GlobalAveragePooling2D
  │  2048-dim vector  (collapses spatial dims, keeps channels)
  ▼
Dense(256)
  │  256-dim vector
  ▼
BatchNormalization → ReLU
  ▼
Dropout(0.5)
  ▼
Dense(6, activation='softmax')
  │  6 class probabilities
  ▼
Output: [cloud, dust, haze, land, seaside, smoke]
```

---

## Hyperparameters

### Architecture

| Parameter | Value | Why |
|---|---|---|
| Input shape | 224 × 224 × 3 | ResNet50's expected input size; our images are already 224×224 |
| Backbone weights | `imagenet` | Pretrained feature extractor — no need to learn edges/textures from scratch |
| `include_top` | `False` | Drop ResNet's original 1000-class head; we add our own 6-class head |
| Pooling after backbone | `GlobalAveragePooling2D` | Converts 7×7×2048 feature map to a 2048-vector — only 2048 values enter the dense head. The alternative, Flatten, would pass all 7×7×2048 = 100,352 values instead, creating far more parameters and overfitting risk |
| Dense head width | 256 | Same as our Custom CNN for comparability; wide enough to learn class boundaries, narrow enough to regularise |
| Dropout rate | 0.5 | Standard for dense heads; randomly zeroes half the neurons each step, forcing redundancy and reducing overfitting |
| Output neurons | 6 | One per class |
| Output activation | `softmax` | Converts raw scores to probabilities that sum to 1 |

### Phase 1 — Head Training (backbone frozen)

| Parameter | Value | Why |
|---|---|---|
| Frozen layers | All ResNet50 layers | Preserve ImageNet weights; only the randomly-initialised head should update |
| Optimizer | Adam | Adaptive learning rates per parameter; converges faster than plain SGD for this task |
| Learning rate | `1e-3` | Standard starting LR for Adam on a fresh head; high enough to learn quickly, low enough not to overshoot |
| Loss | `sparse_categorical_crossentropy` | Labels are integers (0–5), not one-hot vectors |
| Batch size | 32 | Fits comfortably in GPU memory for 224×224 images; gives stable gradient estimates |
| Epochs | 15 (max) | Head converges quickly; EarlyStopping will cut this short if validation loss plateaus |
| EarlyStopping patience | 5 | Stop if val loss hasn't improved for 5 consecutive epochs |

### Phase 2 — Fine-tuning (partial unfreeze)

| Parameter | Value | Why |
|---|---|---|
| Unfrozen layers | Last ~30 layers (Stage 5 + end of Stage 4) | Early layers learn universal features (edges, colours) that transfer well; later layers learn task-specific patterns worth adapting to satellite imagery |
| Optimizer | Adam | Same optimizer, but re-instantiated so momentum doesn't carry over from Phase 1 |
| Learning rate | `1e-5` | 100× smaller than Phase 1 — must be tiny to nudge pretrained weights without catastrophic forgetting |
| Epochs | 15 (max) | Fine-tuning needs more time but also more caution; EarlyStopping guards against overfitting |
| EarlyStopping patience | 5 | Same as Phase 1 |
| ReduceLROnPlateau | factor=0.5, patience=3 | Halves LR if val loss stalls for 3 epochs; helps squeeze out the last improvement |

---

## Why ResNet50 specifically?

- **Well-understood baseline** — one of the most-studied architectures, so results are easy to contextualise
- **Right size for our dataset** — EfficientNet/InceptionV3 are alternatives, but ResNet50 gives a clean residual-learning story for the report
- **Residual connections** — the skip connections mean even if a block learns nothing useful for our domain, the signal passes through unchanged (no degradation problem)

---

## Expected output

After both phases:
- A trained model saved as `resnet50_finetuned.h5`
- Training curves (loss + accuracy) for both phases
- Per-class and macro Precision / Recall / F1 on the test set
- Confusion matrix
- Total training time (Phase 1 + Phase 2 combined)
