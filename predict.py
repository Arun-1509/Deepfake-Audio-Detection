"""
predict.py — Deepfake Audio Detection Inference Script
=======================================================
MARS Open Projects 2026 — Problem Statement 2

Usage:
    # Single file
    python predict.py --input audio.wav

    # Batch (CSV with 'path' column)
    python predict.py --input files.csv --batch

    # Custom model checkpoint
    python predict.py --input audio.wav --checkpoint my_checkpoint.pt

Output:
    Single file → prints result to console
    Batch       → saves predictions to predictions.csv
"""

import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import librosa

# ── Constants (must match training) ──────────────────────────────────────────
SR           = 16000
DURATION     = 4.0
N_SAMPLES    = int(SR * DURATION)
N_MFCC       = 40
HOP_LEN      = 512
N_FFT        = 1024
LAYER_INDICES = [6, 9, 12]
MODEL_NAME   = "facebook/wav2vec2-base"
SUPPORTED_FORMATS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model Architecture (must match training exactly) ─────────────────────────
class DeepfakeDetector(nn.Module):
    def __init__(self, w2v_dim=768, n_acou=191, n_classes=2):
        super().__init__()

        self.w2v_proj = nn.Sequential(
            nn.Linear(w2v_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.45),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.35),
        )

        self.acou_input = nn.Sequential(
            nn.Linear(n_acou, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.45),
        )
        self.acou_res = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.35),
        )

        self.gate = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, w2v_emb, acou_feats):
        w   = self.w2v_proj(w2v_emb)
        a0  = self.acou_input(acou_feats)
        a   = a0 + self.acou_res(a0)
        wts = self.gate(torch.cat([w, a], dim=1))
        fused = wts[:, 0:1] * w + wts[:, 1:2] * a
        return self.classifier(fused), wts


# ── Audio Loading ─────────────────────────────────────────────────────────────
def load_audio(fpath, sr=SR, duration=DURATION):
    """Load, resample, pad/trim, and peak-normalize audio."""
    try:
        audio, _ = librosa.load(fpath, sr=sr, mono=True, duration=duration)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file '{fpath}': {e}")

    target = int(sr * duration)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]

    if np.abs(audio).max() > 0:
        audio = audio / (np.abs(audio).max() + 1e-8)

    return audio.astype(np.float32)


# ── Acoustic Feature Extraction ───────────────────────────────────────────────
def extract_acoustic_features(audio, sr=SR):
    """Extract 191-dim extended acoustic feature vector."""
    feats = []

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LEN)
    feats.append(mfcc.mean(axis=1))            # 40
    feats.append(mfcc.std(axis=1))             # 40

    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feats.append(delta1.mean(axis=1))          # 40
    feats.append(delta2.mean(axis=1))          # 40

    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN)
    feats.append(contrast.mean(axis=1))        # 7

    bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN)
    feats.append(np.array([bw.mean(), bw.std()]))   # 2

    flat = librosa.feature.spectral_flatness(y=audio, n_fft=N_FFT, hop_length=HOP_LEN)
    feats.append(np.array([flat.mean(), flat.std()]))  # 2

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN)
    feats.append(chroma.mean(axis=1))          # 12

    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LEN)
    feats.append(np.array([zcr.mean(), zcr.std()]))    # 2

    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN)
    feats.append(np.array([rolloff.mean(), rolloff.std()]))  # 2

    rms = librosa.feature.rms(y=audio, hop_length=HOP_LEN)
    feats.append(np.array([rms.mean(), rms.std()]))    # 2

    return np.concatenate(feats).astype(np.float32)    # 191 total


# ── Model Loader ──────────────────────────────────────────────────────────────
def load_model(checkpoint_path):
    """Load model, wav2vec2, and config from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: '{checkpoint_path}'")

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    cfg          = ckpt.get("model_config", {})
    n_acou       = cfg.get("n_acou_feats", 191)
    opt_threshold = ckpt.get("opt_threshold", 0.5)
    layer_indices = ckpt.get("layer_indices", LAYER_INDICES)

    model = DeepfakeDetector(w2v_dim=768, n_acou=n_acou).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loading wav2vec2 ({MODEL_NAME}) ...")
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    w2v_model         = Wav2Vec2Model.from_pretrained(MODEL_NAME, output_hidden_states=True).to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    for param in w2v_model.parameters():
        param.requires_grad = False
    w2v_model.eval()

    print(f"✅ Model ready  |  EER-optimal threshold: {opt_threshold:.4f}  |  Device: {device}\n")
    return model, w2v_model, feature_extractor, opt_threshold, layer_indices


# ── Single File Inference ─────────────────────────────────────────────────────
def predict_single(file_path, model, w2v_model, feature_extractor,
                   opt_threshold, layer_indices):
    """Predict a single audio file. Returns result dict."""
    from transformers import Wav2Vec2FeatureExtractor  # already imported above

    ext = os.path.splitext(file_path)[-1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{ext}'. Supported: {SUPPORTED_FORMATS}")

    audio      = load_audio(file_path)
    acou_feats = extract_acoustic_features(audio)

    w2v_input = feature_extractor(
        audio, sampling_rate=SR, return_tensors="pt",
        padding="max_length", max_length=N_SAMPLES, truncation=True
    )

    with torch.no_grad():
        out = w2v_model(w2v_input["input_values"].to(device))
        layer_embs = torch.stack(
            [out.hidden_states[i] for i in layer_indices], dim=-1
        )
        w2v_emb = layer_embs.mean(dim=-1).mean(dim=1)

        logits, gate_w = model(
            w2v_emb,
            torch.tensor(acou_feats, dtype=torch.float32).unsqueeze(0).to(device)
        )
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred   = 1 if probs[1] >= opt_threshold else 0
        gw     = gate_w.cpu().numpy()[0]

    label      = "Deepfake (AI-Generated)" if pred == 1 else "Genuine (Human)"
    confidence = float(probs[pred])

    return {
        "file":                  os.path.basename(file_path),
        "prediction":            label,
        "confidence":            round(confidence, 4),
        "prob_genuine":          round(float(probs[0]), 4),
        "prob_deepfake":         round(float(probs[1]), 4),
        "threshold_used":        round(float(opt_threshold), 4),
        "gate_weight_wav2vec2":  round(float(gw[0]), 4),
        "gate_weight_acoustic":  round(float(gw[1]), 4),
    }


# ── Pretty Print ──────────────────────────────────────────────────────────────
def print_result(result):
    is_fake = "Deepfake" in result["prediction"]
    verdict = "🔴 DEEPFAKE" if is_fake else "🟢 GENUINE"

    print("=" * 55)
    print(f"  FILE       : {result['file']}")
    print(f"  VERDICT    : {verdict}")
    print(f"  CONFIDENCE : {result['confidence']*100:.2f}%")
    print(f"  Prob Genuine  : {result['prob_genuine']*100:.2f}%")
    print(f"  Prob Deepfake : {result['prob_deepfake']*100:.2f}%")
    print(f"  Threshold     : {result['threshold_used']}")
    print(f"  Gate [wav2vec2 / acoustic] : "
          f"{result['gate_weight_wav2vec2']:.3f} / {result['gate_weight_acoustic']:.3f}")
    print("=" * 55)


# ── Batch Inference ───────────────────────────────────────────────────────────
def predict_batch(csv_path, model, w2v_model, feature_extractor,
                  opt_threshold, layer_indices, output_csv="predictions.csv"):
    """Run inference on all files listed in a CSV (must have a 'path' column)."""
    import pandas as pd
    from tqdm import tqdm

    df = pd.read_csv(csv_path)
    if "path" not in df.columns:
        raise ValueError("CSV must have a 'path' column with audio file paths.")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        fpath = row["path"]
        try:
            res = predict_single(fpath, model, w2v_model, feature_extractor,
                                 opt_threshold, layer_indices)
            res["true_label"] = row.get("label", "unknown")
            results.append(res)
        except Exception as e:
            results.append({
                "file": os.path.basename(fpath),
                "prediction": "ERROR",
                "confidence": 0.0,
                "error": str(e)
            })

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"\n✅ Predictions saved to: {output_csv}")
    print(f"   Total files: {len(results)}")
    if "prediction" in out_df.columns:
        print(f"   Genuine:  {(out_df['prediction'].str.contains('Genuine')).sum()}")
        print(f"   Deepfake: {(out_df['prediction'].str.contains('Deepfake')).sum()}")
        print(f"   Errors:   {(out_df['prediction'] == 'ERROR').sum()}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Deepfake Audio Detection — Inference Script",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to an audio file (.wav/.flac/.mp3/.ogg/.m4a)\n"
             "or a CSV file with a 'path' column (use with --batch)"
    )
    parser.add_argument(
        "--checkpoint", default="deepfake_checkpoint.pt",
        help="Path to model checkpoint (default: deepfake_checkpoint.pt)"
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run batch inference on a CSV file"
    )
    parser.add_argument(
        "--output", default="predictions.csv",
        help="Output CSV path for batch mode (default: predictions.csv)"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override EER-optimal threshold (default: use checkpoint value)"
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Load model
    model, w2v_model, feature_extractor, opt_thr, layer_indices = \
        load_model(args.checkpoint)

    # Override threshold if provided
    if args.threshold is not None:
        print(f"⚠️  Threshold overridden: {opt_thr:.4f} → {args.threshold:.4f}")
        opt_thr = args.threshold

    if args.batch:
        # Batch mode
        predict_batch(
            args.input, model, w2v_model, feature_extractor,
            opt_thr, layer_indices, output_csv=args.output
        )
    else:
        # Single file mode
        print(f"\nAnalyzing: {args.input}")
        result = predict_single(
            args.input, model, w2v_model, feature_extractor,
            opt_thr, layer_indices
        )
        print_result(result)


if __name__ == "__main__":
    main()
