# Deepfake Audio Detection
### MARS Open Projects 2026 — Problem Statement 2

A hybrid deep learning system that classifies speech recordings as **Genuine (Human)** or **Deepfake (AI-Generated)** with 90.89% accuracy and 9.11% EER on the Fake-or-Real dataset.

---

## Results

| Metric | Required | Achieved |
|---|---|---|
| Overall Accuracy | ≥ 80% | **90.89%** ✅ |
| EER | ≤ 12% | **9.11%** ✅ |
| Macro F1 | ≥ 80% | **90.89%** ✅ |
| Accuracy — Genuine | ≥ 75% | **90.90%** ✅ |
| Accuracy — Deepfake | ≥ 75% | **90.89%** ✅ |
| ROC-AUC | — | **0.931** |

---

## Project Structure

```
deepfake-audio-detection/
├── Deepfake_Audio_Detection.ipynb   # Full pipeline: preprocessing → training → evaluation
├── predict.py                       # Standalone inference script for new audio files
├── deepfake_checkpoint.pt           # Trained model checkpoint (weights + config + threshold)
├── deepfake_model.pt                # Model weights only
├── performance_report.json          # Full metrics report (accuracy, EER, F1, AUC)
├── deepfake_results.png             # Confusion matrix, ROC curve, training history
├── requirements.txt                 # Pinned dependencies
└── README.md
```

---

## Model Architecture

**Dual-Stream MLP + Attention Gate**

Two parallel feature streams are fused through a learned attention gate that dynamically weights each stream per sample:

**Stream 1 — wav2vec2-base (frozen)**
- Pre-trained `facebook/wav2vec2-base` with all weights frozen
- Hidden states from layers 6, 9, and 12 are extracted and averaged → 768-dim embedding
- Multi-layer averaging gives richer representations than the last layer alone

**Stream 2 — Extended Acoustic Features (191-dim)**
- MFCC mean (40) + MFCC std (40)
- Delta-MFCC (40) + Delta2-MFCC (40)
- Spectral Contrast (7)
- Spectral Bandwidth (2) + Spectral Flatness (2)
- Chroma (12)
- Zero Crossing Rate (2)
- Spectral Rolloff (2)
- RMS Energy (2)

**Fusion — Attention Gate**
- Concatenates both stream outputs → 512-dim
- Learns per-sample soft weights (Softmax) for each stream
- Fused representation passed to classifier MLP

**Classifier MLP**
```
256 → BatchNorm → GELU → Dropout(0.4)
→ 128 → BatchNorm → GELU → Dropout(0.4)
→ 64 → GELU → Dropout(0.3)
→ 2 (Genuine / Deepfake)
```

---

## Preprocessing

- **Sample rate:** 16,000 Hz (resampled if different)
- **Duration:** 4 seconds (padded with zeros if shorter, trimmed if longer)
- **Normalization:** Peak normalization to [-1, 1]
- **wav2vec2 features:** Precomputed once before training for speed

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 2e-4 |
| Weight Decay | 0.08 |
| Scheduler | OneCycleLR (cosine annealing) |
| Batch Size | 256 |
| Max Epochs | 25 |
| Early Stopping | Patience = 8 (on EER) |
| Label Smoothing | 0.08 |
| Dropout | 0.35–0.45 |
| Gradient Clipping | 1.0 |

**Regularization:** BatchNorm + Dropout 0.35–0.45 + weight decay 0.08 + gradient clipping

**Class Imbalance:** Handled via class-weighted CrossEntropyLoss (sklearn `compute_class_weight`)

**Augmentation:**
- Genuine: Gaussian noise, time stretch, pitch shift, random shift
- Deepfake: Light Gaussian noise, random shift (lighter to preserve codec artifacts)

**Inference threshold:** EER-optimal threshold (0.042) instead of naive 0.5 — directly minimizes EER

---

## Pipeline

```
1. Download dataset (for-norm split from Fake-or-Real dataset)
2. Build file list → stratified train/val/test split
3. Extract 191-dim acoustic features (MFCC, delta, spectral, chroma, ZCR, RMS)
4. Precompute wav2vec2-base embeddings (layers 6, 9, 12 averaged) — cached for fast training
5. Build DeepfakeDetector (dual-stream MLP + attention gate)
6. Train with AdamW + OneCycleLR + early stopping on EER
7. Evaluate on test set with EER-optimal threshold
8. Generate confusion matrix, ROC curve, training history plots
9. Export performance_report.json
10. Save model checkpoint
```

---

## Dataset

**The Fake-or-Real Dataset**
- Link: [kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)
- Split used: `for-norm/train` directory
- Classes: `real/` (Genuine) and `fake/` (Deepfake)

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Deepfake-Audio-Detection.git
cd deepfake-audio-detection

# Install dependencies
pip install -r requirements.txt
```

---

## Running Inference

**Single audio file:**
```bash
python predict.py --input path/to/audio.wav
```

**Example output:**
```
==================================================
  FILE       : audio.wav
  VERDICT    : 🔴 DEEPFAKE
  CONFIDENCE : 94.32%
  Prob Genuine  : 5.68%
  Prob Deepfake : 94.32%
  Threshold     : 0.042
  Gate [wav2vec2 / acoustic] : 0.61 / 0.39
==================================================
```

**Batch inference (CSV with `path` column):**
```bash
python predict.py --input files.csv --batch --output predictions.csv
```

**Custom checkpoint or threshold:**
```bash
python predict.py --input audio.wav --checkpoint deepfake_checkpoint.pt --threshold 0.5
```

---

## Performance Report

Full metrics are saved in `performance_report.json`. Key results:

```json
{
  "metrics": {
    "overall_accuracy": 0.9089,
    "eer": 0.0911,
    "macro_f1": 0.9089,
    "accuracy_genuine": 0.9090,
    "accuracy_deepfake": 0.9089,
    "roc_auc": 0.931
  },
  "inference": {
    "threshold": "EER-optimal (not fixed 0.5)",
    "optimal_threshold": 0.042
  }
}
```

Confusion matrix, ROC curve, and training history plots are saved in `deepfake_results.png`.

---

## Author

**Arun Kaarthikeyan R — 23321006**
MARS Open Projects 2026 | Machine Learning & Deep Learning
