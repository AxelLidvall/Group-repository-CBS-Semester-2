# CLAUDE.md

CBS 8th semester ML/DL exam. Goal: build and compare three models (Custom CNN, Transfer Learning, SVM) for 6-class remote sensing image classification on the USTC_SmokeRS dataset, then write an academic report.

## Notebook style

The active notebook is `main_project_v2.ipynb`. Keep it clean and readable:
- No excessive markdown headers or section dividers inside cells
- Each code cell does one clear thing — no giant monolithic cells
- Minimal inline comments; let the code speak
- The user will ask questions about each step after it is implemented — write code that is easy to explain

## Imports

There is a single "Importing Libraries" cell in the notebook (cell id `b35350e7`). All imports live there.

Before adding any import statement:
1. Read cell `b35350e7` to check what is already imported
2. If the library is already imported, do not add it again
3. If it is new, add it to that cell — never add import statements to any other cell

## Workflow

We follow `checklist.md` step by step, one item at a time. The user will ask questions after each step — wait for the go-ahead before moving on.

**After implementing any checklist item**, immediately tick the corresponding box(es) in `checklist.md` by changing `[ ]` to `[x]`. Do this every time, without being asked.

## Running code

```bash
jupyter notebook main_project_v2.ipynb
```

Dependencies: `numpy pandas matplotlib scikit-learn Pillow datasets tifffile torch torchvision`

## Dataset

**USTC_SmokeRS** — HuggingFace `jonathan-roberts1/USTC_SmokeRS`
- 6,225 RGB satellite images, 256×256 px, uint8
- 6 classes: `cloud`, `dust`, `haze`, `land`, `seaside`, `smoke`
- Classes are roughly balanced (~1,000–1,164 images each) — no severe imbalance
- Loaded via `load_dataset("jonathan-roberts1/USTC_SmokeRS", split="train")`
- Images come as PIL objects — convert to numpy before processing

**Pipeline:**
1. Load dataset from HuggingFace
2. Resize all images to 224×224
3. Fresh stratified 60/20/20 train/val/test split
4. Augmentation on training split only (expansion only — no oversampling needed)

## Normalisation (per model)

- Custom CNN and SVM: divide by 255 → [0, 1]
- Transfer Learning: ImageNet mean [0.485, 0.456, 0.406] / std [0.229, 0.224, 0.225]
- Normalisation functions are defined in 4.2.4 and applied inside each model's training pipeline

## Models

1. **Custom CNN** — from scratch; conv/pool/batchnorm/dropout/dense head (6-class output); Grad-CAM visualisation
2. **Transfer Learning** — ResNet50 / EfficientNet / InceptionV3; freeze → train head → unfreeze fine-tune; 6-class output
3. **SVM** — flattened pixels or CNN features; PCA; grid search with 5-fold CV; multi-class (one-vs-rest)

Evaluation for all three: per-class and macro-average Precision / Recall / F1, confusion matrix, training time.

## Constraints

- Val and test sets are never augmented
- Use only the RGB USTC_SmokeRS dataset
- FASDD_RS is no longer part of this project
- **Do not remove data** — if duplicates, corrupt images, or RGB anomalies are found, flag and report them only; do not drop them from the splits
- **No cropping augmentation** — augmentations are: flip, rotate, brightness, Gaussian noise only
