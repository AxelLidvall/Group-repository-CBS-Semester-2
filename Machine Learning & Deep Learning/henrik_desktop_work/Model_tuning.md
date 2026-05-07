# Custom CNN — Model Tuning

## Baseline

| | |
|---|---|
| **Architecture** | 3 blocks: Conv(32)×2 → Conv(64)×2 → Conv(128)×2, each with BN + ReLU + MaxPool + Dropout(0.25); GAP → Dense(256) → BN → ReLU → Dropout(0.5) → Dense(6, softmax) |
| **Params** | 324,390 |
| **Optimizer** | AdamW lr=1e-3 |
| **Callbacks** | EarlyStopping(patience=7), ReduceLROnPlateau(patience=3, factor=0.5) |
| **Batch / Epochs** | 32 / max 50 |
| **Val accuracy** | ~70% |
| **Test accuracy** | 69% |
| **Macro F1** | 0.68 |

---

## Fixed across all variants
- Input: 224×224×3, normalised /255 → [0, 1]
- Training set: 18,675 augmented images (5× original)
- Val / test sets: never augmented
- Optimizer: AdamW lr=1e-3 (unless LR is the thing being varied)
- Callbacks: same EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
- Batch: 32, max 50 epochs
- Evaluation: `classification_report` — macro F1 is primary metric

---

## Variants to try

### V1 — 4th Block
Try to add another block. So that there are 4. Keep the same structure in the blocks as when there was 3.

---

### V2 — Two-Layer Head
Add more layers to the Dense head to make it a better neural network. Maybe instead of having 256 neurons in 1 layer, try to splitt the neurons over more layers. Or even just add more. Make this decision based on literatur.

---

### V3 — Wide Dense
Try to run the model with just 1 layer again, but increase the number of neurons. We are not that afraid of running time, so maybe try 1024? (if this is way to overkill try 512 instead)

---

### V4 — Half-Width
Try halving (16, 32, 64) for a lighter model.

---

### V5 — Double-Width
Try doubling the Conv2D to (64, 128, 256) for a more capable but heavier model.

---

## Results summary

| Version | Change | Val acc | Test acc | Macro F1 | Params | Notes |
|---------|--------|---------|----------|----------|--------|-------|
| Baseline | 3 blocks 32/64/128, Dense(256) | 70.0% | 69% | 0.68 | 324K | |
| V1 — 4th Block | 4 blocks 32/64/128/256, Dense(256) | 65.6% | 68% | 0.68 | 1,244K | 6.5 min — no gain over baseline |
| V2 — Two-Layer Head | 3 blocks, Dense(512)→Dense(256) | **70.5%** | 70% | 0.69 | 491K | 5.4 min — best val and test accuracy |
| V3 — Wide Dense | 3 blocks, Dense(1024) | 68.3% | 70% | 0.69 | 431K | 4.7 min — tied test acc but lower val |
| V4 — Half-Width | 3 blocks 16/32/64, Dense(256) | 63.4% | 61% | 0.58 | 92K | 2.2 min — too little capacity |
| V5 — Double-Width | 3 blocks 64/128/256, Dense(256) | 67.9% | 69% | 0.68 | 1,217K | 14.2 min — heavy, no gain |

---

## Winner
**Chosen version:** V2 — Two-Layer Head  
**Reason:** Highest validation accuracy (70.5%) — the correct metric for model selection. The test set was only consulted after the winner was identified, and it confirmed the result: V2 achieved 70% test accuracy and 0.69 macro F1. V3 tied on test accuracy but scored lower on the validation set (68.3%), making V2 the methodologically clean choice.


  ┌─────────────────────┬─────────┬──────────┬──────────┬────────┬──────────┐
  │                     │ Val acc │ Test acc │ Macro F1 │ Params │   Time   │
  ├─────────────────────┼─────────┼──────────┼──────────┼────────┼──────────┤
  │ Baseline            │ 70.0%   │ 69%      │ 0.68     │ 324K   │ 7.4 min  │
  ├─────────────────────┼─────────┼──────────┼──────────┼────────┼──────────┤
  │ V1 — 4th Block      │ 65.6%   │ 68%      │ 0.68     │ 1,244K │ 6.5 min  │
  ├─────────────────────┼─────────┼──────────┼──────────┼────────┼──────────┤
  │ V2 — Two-Layer Head │ 70.5% ★ │ 70%      │ 0.69     │ 491K   │ 5.4 min  │
  ├─────────────────────┼─────────┼──────────┼──────────┼────────┼──────────┤
  │ V3 — Wide Dense     │ 68.3%   │ 70%      │ 0.69     │ 431K   │ 4.7 min  │
  ├─────────────────────┼─────────┼──────────┼──────────┼────────┼──────────┤
  │ V4 — Half-Width     │ 63.4%   │ 61%      │ 0.58     │ 92K    │ 2.2 min  │
  ├─────────────────────┼─────────┼──────────┼──────────┼────────┼──────────┤
  │ V5 — Double-Width   │ 67.9%   │ 69%      │ 0.68     │ 1,217K │ 14.2 min │
  └─────────────────────┴─────────┴──────────┴──────────┴────────┴──────────┘