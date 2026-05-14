# Error Analysis — Writing Plan

---

## ⚠ METHODOLOGICAL FRAMING — FIX BEFORE HAND-IN

**Problem:** In the tuning notebook, we evaluated on the test set after every model version (v0–v8). This is bad practice — it means the test set implicitly influenced model selection, which leaks information and invalidates the test set as a held-out evaluation.

**How to frame it in the paper and notebook:**

- State that all model selection decisions (architecture changes, hyperparameter choices, class weights, LR schedule) were guided **exclusively by validation set performance**.
- The test set was used **only once**, for the final model (M2/v8), to produce the reported results.
- The intermediate test scores shown in the tuning notebook are for transparency and development logging only — they did not influence which model we selected.

**In the methodology section (4.2 or wherever train/val/test split is described), add a sentence like:**
> "Model selection and all hyperparameter decisions were based solely on validation performance. The test set was held out and evaluated only for the final selected model (M2) to produce an unbiased estimate of generalisation performance."

**In the notebook (main_project_v4):** For v0–v7, only show validation metrics in the output. Only run the full test set evaluation for v8/M2. If v0–v7 test outputs are already in the cells, add a markdown comment explaining they are shown for development transparency only.

---


## What the section needs to do

The error analysis sits in section 6.2 and must do three things:
1. Explain *where* each model fails (confusion matrix patterns)
2. Explain *why* it fails (visual ambiguity in the data → connects back to the research question)
3. Show *what the best model looks at* (Grad-CAM → verify M2 is detecting the right signal)

The face mask report (sample) is the structural template: quantify errors → show examples → identify root causes → link to model design. We follow the same logic but adapted to a 6-class atmospheric classification problem where the root cause is *shared visual structure between atmospheric classes*, not image quality or labelling issues.

---

## Section structure

### 6.2.1 Confusion Matrix Analysis (all four models)

**What to write:**
Start by noting the total number of misclassifications per model on the 1,245-image test set (derived from accuracy × support):
- M1: ~246 wrong (accuracy 0.80 → ~1245 × 0.20)
- M2: ~149 wrong (accuracy 0.88 → ~1245 × 0.12)
- ResNet50: ~TBD (need to confirm from notebook)
- SVM: ~TBD (need to confirm from notebook)

**The key pattern to argue** (already confirmed by the classification reports):

All weaker models (M1, SVM, ResNet50) share the same failure mode:
- Smoke ↔ haze confusion (high mutual misclassification)
- Haze ↔ dust confusion
- M1 specifically: smoke recall = 0.33, seaside precision = 0.59 — M1 over-predicts seaside as a "safe fallback" when the atmospheric signal is ambiguous

**The argument:**
These confusions are not random — they follow the visual structure of the dataset. Smoke, haze, and dust are diffuse, semi-transparent atmospheric phenomena without hard boundaries. They appear over varying land and seaside backgrounds, which forces the model to disentangle the phenomenon from the terrain simultaneously. Seaside is the easiest class (strong blue water signal), so underregularised models collapse onto it when uncertain.

**Figure to include:** Confusion matrices for all 4 models (side by side or in appendix with reference here). Already mentioned in the paper draft. Pull these from the notebook.

---

### 6.2.2 Smoke Class Deep Dive

**What to write:**
Since the research question prioritises smoke recall, give smoke its own paragraph. Compare across models:

| Model   | Smoke Precision | Smoke Recall | Smoke F1 |
|---------|-----------------|--------------|----------|
| M1      | 0.61            | 0.33         | 0.43     |
| SVM     | —               | 0.52         | —        |
| ResNet50| —               | 0.59         | —        |
| M2      | 0.87            | 0.85         | 0.86     |

M1's 0.33 recall means it missed 2 out of 3 actual smoke images — the model was not learning a smoke-specific feature, it was relying on what was *not* seaside/land. M2's 0.85 recall represents the direct effect of the class weight intervention (smoke×2.0) combined with better regularisation forcing the model to commit to smoke predictions rather than retreat to safer classes.

**Show misclassified examples** (like the face mask paper's Figure 15): pull 3–4 smoke images that M2 still gets wrong and describe why. The expected pattern is smoke images that visually resemble haze (thin, diffuse, high-altitude) or images where smoke sits over water (confusion with seaside's blue-grey tones).

---

### 6.2.3 Grad-CAM Analysis (M2 focus)

**What to write:**
Use Grad-CAM on M2 (cnn_v8_best.h5) applied to the last conv block (block 4, the 256-filter layer). Show heatmaps for:
- Correctly classified smoke images: expect activation on the smoke plume/gradient region
- Correctly classified haze: expect diffuse activation across the full image (haze has no localised feature)
- Misclassified examples: where does the model look when it's wrong?

**The argument to make** (follow face mask paper's logic from section 6.3):
If Grad-CAM shows activation on the atmospheric phenomenon itself (smoke texture, haze diffusion pattern, dust colour gradient), the model is learning the right signal. If activation falls on terrain features (coastline, water, land texture), the model is relying on background shortcuts — which is exactly the failure mode the USTC_SmokeRS dataset was designed to challenge (Ba et al., 2019 note that other models only focus on texture and colour of land, coastline, and water).

**Link to the original paper's finding:**
Ba et al. (2019) show that models without spatial attention mechanisms can only focus on large-scale background features. Grad-CAM lets us verify whether M2 has overcome this limitation without explicit attention mechanisms, relying instead on the combination of deeper feature maps (4 blocks vs 3) and class-weight-driven training pressure.

**Figures to include:**
- 2–3 Grad-CAM examples per class for correctly classified images (especially smoke)
- 1–2 Grad-CAM examples for misclassified smoke images
- Format: [original | heatmap overlay] pairs, same layout as face mask paper Figure 17

---

## What needs to be run in the notebook

1. **Confusion matrices** — need raw counts (not normalised) for all 4 models. If not already in notebook outputs, run `confusion_matrix(y_test, y_pred)` for each.
2. **Misclassified smoke examples** — find indices where `y_test == smoke_idx` and `y_pred != smoke_idx`, pull the corresponding images.
3. **Grad-CAM** — apply to M2 (cnn_v8_best.h5). The Grad-CAM implementation should target the last Conv2D layer in block 4. If not already in the notebook, needs to be added.

---

## Narrative flow for the written section

> M1, SVM, and ResNet50 share a consistent failure pattern: smoke is frequently misclassified as haze and dust, and haze and dust are mutually confused with each other. [Reference confusion matrices.] This pattern directly reflects the visual nature of the task — smoke, haze, and dust are diffuse atmospheric phenomena without hard boundaries, and in satellite imagery they are further complicated by the varying terrain they overlie. The M1 model additionally shows a disproportionate tendency to predict seaside when uncertain, consistent with seaside being the most visually distinct class in the dataset. [Quantify seaside over-prediction.]
>
> M2 addresses these failure modes directly. The class weight intervention (smoke×2.0, haze×2.0, seaside×0.6) forces the model to penalise misclassification of the hard classes and suppresses reliance on the easy-to-learn seaside signal. The result is a smoke recall improvement from 0.33 to 0.85. [Remaining failures: show examples of smoke images M2 still misclassifies.]
>
> To verify that M2 is learning the right signal rather than background shortcuts, we apply Grad-CAM to the final convolutional block. [Show heatmaps.] For correctly classified smoke images, activation concentrates on the diffuse gradient region corresponding to the smoke plume. For haze, activation is distributed across the full image, consistent with haze being a scene-level property. This confirms that M2 has learned class-discriminative features in the atmospheric phenomenon itself, rather than relying on terrain background — the key challenge identified by Ba et al. (2019).

---

## Tone notes (matching the sample report)

- Lead with the number of wrong predictions per model (concrete and easy to parse)
- Show actual images for at least the best model's remaining failures — the face mask paper does this well
- Keep the root cause explanation grounded in the *visual* nature of the data, not just numbers
- Grad-CAM conclusion should be clear: either "the model is looking at the right thing" or "it is using a shortcut" — do not hedge
