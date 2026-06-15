"""
app.py — Deepfake Audio Detection | MARS Open Projects 2026
Streamlit Web App
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎙️",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background-color: #0e0e12;
    color: #e8e8f0;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }

/* Hero */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #a5b4fc;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 100px;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    color: #f0f0ff;
    letter-spacing: -0.02em;
    line-height: 1.15;
    margin: 0 0 0.6rem 0;
}
.hero-sub {
    font-size: 1rem;
    color: #8888aa;
    font-weight: 400;
    margin: 0;
}

/* Upload box */
.upload-section {
    background: #16161e;
    border: 1.5px dashed #2e2e44;
    border-radius: 14px;
    padding: 2rem 1.5rem;
    text-align: center;
    margin: 1.5rem 0;
    transition: border-color 0.2s;
}

/* Verdict cards */
.verdict-genuine {
    background: linear-gradient(135deg, #0d2818 0%, #0a1f12 100%);
    border: 1.5px solid #22c55e;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
}
.verdict-deepfake {
    background: linear-gradient(135deg, #2a0a0a 0%, #1f0808 100%);
    border: 1.5px solid #ef4444;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
}
.verdict-icon {
    font-size: 3rem;
    margin-bottom: 0.5rem;
}
.verdict-label {
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    margin: 0.2rem 0;
}
.verdict-genuine .verdict-label { color: #22c55e; }
.verdict-deepfake .verdict-label { color: #ef4444; }
.verdict-conf {
    font-size: 0.95rem;
    color: #8888aa;
    margin-top: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Metric cards */
.metrics-row {
    display: flex;
    gap: 0.8rem;
    margin: 1rem 0;
}
.metric-card {
    flex: 1;
    background: #16161e;
    border: 1px solid #2e2e44;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #a5b4fc;
}
.metric-label {
    font-size: 0.72rem;
    color: #666688;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* Section headers */
.section-header {
    font-size: 0.72rem;
    font-weight: 600;
    color: #555577;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1.8rem 0 0.8rem 0;
    border-bottom: 1px solid #1e1e2a;
    padding-bottom: 0.5rem;
}

/* Info box */
.info-box {
    background: #16161e;
    border: 1px solid #2e2e44;
    border-left: 3px solid #6366f1;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    color: #9999bb;
    margin: 1rem 0;
}

/* Gate weights */
.gate-bar-container {
    margin: 0.6rem 0;
}
.gate-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: #8888aa;
    margin-bottom: 0.3rem;
}
.gate-bar-bg {
    background: #1e1e2a;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
    margin-bottom: 0.6rem;
}
.gate-bar-fill-w2v {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #6366f1, #818cf8);
}
.gate-bar-fill-acou {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #06b6d4, #22d3ee);
}

/* Footer */
.footer {
    text-align: center;
    color: #444466;
    font-size: 0.78rem;
    padding: 2.5rem 0 1rem 0;
    border-top: 1px solid #1e1e2a;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SR            = 16000
DURATION      = 4.0
N_SAMPLES     = int(SR * DURATION)
N_MFCC        = 40
HOP_LEN       = 512
N_FFT         = 1024
LAYER_INDICES = [6, 9, 12]
MODEL_NAME    = "facebook/wav2vec2-base"
CHECKPOINT    = "deepfake_checkpoint.pt"


# ── Model Architecture ────────────────────────────────────────────────────────
class DeepfakeDetector(nn.Module):
    def __init__(self, w2v_dim=768, n_acou=191, n_classes=2):
        super().__init__()
        self.w2v_proj = nn.Sequential(
            nn.Linear(w2v_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.45),
            nn.Linear(512, 256),    nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.35),
        )
        self.acou_input = nn.Sequential(
            nn.Linear(n_acou, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.45),
        )
        self.acou_res = nn.Sequential(
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.35),
        )
        self.gate = nn.Sequential(
            nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 2), nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(128, 64),  nn.GELU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, w2v_emb, acou_feats):
        w   = self.w2v_proj(w2v_emb)
        a0  = self.acou_input(acou_feats)
        a   = a0 + self.acou_res(a0)
        wts = self.gate(torch.cat([w, a], dim=1))
        return self.classifier(wts[:, 0:1] * w + wts[:, 1:2] * a), wts


# ── Load Model (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt          = torch.load(CHECKPOINT, map_location=device)
    cfg           = ckpt.get("model_config", {})
    n_acou        = cfg.get("n_acou_feats", 191)
    opt_threshold = ckpt.get("opt_threshold", 0.5)
    layer_indices = ckpt.get("layer_indices", LAYER_INDICES)

    model = DeepfakeDetector(w2v_dim=768, n_acou=n_acou).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    w2v       = Wav2Vec2Model.from_pretrained(MODEL_NAME, output_hidden_states=True).to(device)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    for p in w2v.parameters(): p.requires_grad = False
    w2v.eval()

    return model, w2v, extractor, opt_threshold, layer_indices, device


# ── Audio Processing ──────────────────────────────────────────────────────────
def load_audio(file_bytes):
    import soundfile as sf
    import io
    audio, sr = sf.read(io.BytesIO(file_bytes))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SR)
    target = int(SR * DURATION)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]
    if np.abs(audio).max() > 0:
        audio = audio / (np.abs(audio).max() + 1e-8)
    return audio.astype(np.float32)


def extract_acoustic_features(audio):
    feats = []
    mfcc  = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LEN)
    feats += [mfcc.mean(axis=1), mfcc.std(axis=1)]
    feats += [librosa.feature.delta(mfcc).mean(axis=1),
              librosa.feature.delta(mfcc, order=2).mean(axis=1)]
    feats.append(librosa.feature.spectral_contrast(y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LEN).mean(axis=1))
    bw   = librosa.feature.spectral_bandwidth(y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LEN)
    flat = librosa.feature.spectral_flatness(y=audio, n_fft=N_FFT, hop_length=HOP_LEN)
    feats += [np.array([bw.mean(), bw.std()]), np.array([flat.mean(), flat.std()])]
    feats.append(librosa.feature.chroma_stft(y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LEN).mean(axis=1))
    zcr     = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LEN)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LEN)
    rms     = librosa.feature.rms(y=audio, hop_length=HOP_LEN)
    feats  += [np.array([zcr.mean(), zcr.std()]),
               np.array([rolloff.mean(), rolloff.std()]),
               np.array([rms.mean(), rms.std()])]
    return np.concatenate(feats).astype(np.float32)


def predict(audio, model, w2v, extractor, opt_threshold, layer_indices, device):
    acou = extract_acoustic_features(audio)
    inp  = extractor(audio, sampling_rate=SR, return_tensors="pt",
                     padding="max_length", max_length=N_SAMPLES, truncation=True)
    with torch.no_grad():
        out       = w2v(inp["input_values"].to(device))
        layer_embs = torch.stack([out.hidden_states[i] for i in layer_indices], dim=-1)
        w2v_emb   = layer_embs.mean(dim=-1).mean(dim=1)
        logits, gate_w = model(
            w2v_emb,
            torch.tensor(acou, dtype=torch.float32).unsqueeze(0).to(device)
        )
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        gw    = gate_w.cpu().numpy()[0]
    pred  = 1 if probs[1] >= opt_threshold else 0
    return pred, probs, gw


# ── Waveform Plot ─────────────────────────────────────────────────────────────
def plot_waveform(audio, is_deepfake):
    color = "#ef4444" if is_deepfake else "#22c55e"
    fig, ax = plt.subplots(figsize=(8, 1.8))
    fig.patch.set_facecolor("#16161e")
    ax.set_facecolor("#16161e")
    times = np.linspace(0, DURATION, len(audio))
    ax.fill_between(times, audio, alpha=0.6, color=color)
    ax.plot(times, audio, color=color, linewidth=0.5, alpha=0.9)
    ax.axhline(0, color="#2e2e44", linewidth=0.8)
    ax.set_xlim(0, DURATION)
    ax.set_xlabel("Time (s)", color="#555577", fontsize=8)
    ax.tick_params(colors="#555577", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2e2e44")
    plt.tight_layout(pad=0.3)
    return fig


# ── Probability Bar Chart ─────────────────────────────────────────────────────
def plot_probs(probs):
    fig, ax = plt.subplots(figsize=(8, 1.6))
    fig.patch.set_facecolor("#16161e")
    ax.set_facecolor("#16161e")
    labels = ["Genuine", "Deepfake"]
    colors = ["#22c55e", "#ef4444"]
    bars   = ax.barh(labels, [p * 100 for p in probs], color=colors,
                     height=0.4, alpha=0.85)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{p*100:.1f}%", va="center", color="#e8e8f0", fontsize=9,
                fontfamily="monospace")
    ax.set_xlim(0, 115)
    ax.set_xlabel("Probability (%)", color="#555577", fontsize=8)
    ax.tick_params(colors="#aaaacc", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2e2e44")
    plt.tight_layout(pad=0.3)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-badge">MARS Open Projects 2026</div>
    <h1 class="hero-title">Deepfake Audio Detector</h1>
    <p class="hero-sub">Upload a speech recording to detect whether it is human or AI-generated</p>
</div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("Loading model..."):
    try:
        model, w2v, extractor, opt_threshold, layer_indices, device = load_model()
        st.markdown("""
        <div class="info-box">
            ✅ &nbsp; Model ready &nbsp;·&nbsp; Dual-Stream MLP + Attention Gate &nbsp;·&nbsp;
            wav2vec2-base + 191-dim acoustic features &nbsp;·&nbsp; EER: 9.11%
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# Upload
st.markdown('<div class="section-header">Upload Audio</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Supported formats: WAV, FLAC, MP3, OGG",
    type=["wav", "flac", "mp3", "ogg"],
    label_visibility="visible"
)

if uploaded is not None:
    file_bytes = uploaded.read()

    # Playback
    st.markdown('<div class="section-header">Audio Playback</div>', unsafe_allow_html=True)
    st.audio(file_bytes, format=f"audio/{uploaded.name.split('.')[-1]}")

    # Run inference
    with st.spinner("Analysing audio..."):
        try:
            audio          = load_audio(file_bytes)
            pred, probs, gw = predict(audio, model, w2v, extractor,
                                      opt_threshold, layer_indices, device)
            is_deepfake    = pred == 1
            confidence     = float(probs[pred])
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    # ── Verdict ──────────────────────────────────────────────────────────────
    if is_deepfake:
        st.markdown(f"""
        <div class="verdict-deepfake">
            <div class="verdict-icon">🔴</div>
            <div class="verdict-label">Deepfake (AI-Generated)</div>
            <div class="verdict-conf">Confidence: {confidence*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-genuine">
            <div class="verdict-icon">🟢</div>
            <div class="verdict-label">Genuine (Human)</div>
            <div class="verdict-conf">Confidence: {confidence*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-card">
            <div class="metric-value">{probs[0]*100:.1f}%</div>
            <div class="metric-label">Prob Genuine</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{probs[1]*100:.1f}%</div>
            <div class="metric-label">Prob Deepfake</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{opt_threshold:.3f}</div>
            <div class="metric-label">EER Threshold</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Waveform ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Waveform</div>', unsafe_allow_html=True)
    st.pyplot(plot_waveform(audio, is_deepfake), use_container_width=True)

    # ── Probability Chart ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Class Probabilities</div>', unsafe_allow_html=True)
    st.pyplot(plot_probs(probs), use_container_width=True)

    # ── Gate Weights ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Attention Gate Weights</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="gate-bar-container">
        <div class="gate-label">
            <span>wav2vec2-base (deep features)</span>
            <span>{gw[0]*100:.1f}%</span>
        </div>
        <div class="gate-bar-bg">
            <div class="gate-bar-fill-w2v" style="width:{gw[0]*100:.1f}%"></div>
        </div>
        <div class="gate-label">
            <span>Acoustic features (MFCC, spectral, chroma)</span>
            <span>{gw[1]*100:.1f}%</span>
        </div>
        <div class="gate-bar-bg">
            <div class="gate-bar-fill-acou" style="width:{gw[1]*100:.1f}%"></div>
        </div>
    </div>
    <div class="info-box">
        The attention gate dynamically weights both streams per sample.
        Higher wav2vec2 weight means the model relied more on deep speech representations.
        Higher acoustic weight means spectral/prosodic features drove the decision.
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="upload-section">
        <p style="color:#555577; font-size:0.9rem; margin:0">
            Drop an audio file above to begin analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Model Info ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
st.markdown("""
<div class="metrics-row">
    <div class="metric-card">
        <div class="metric-value">90.89%</div>
        <div class="metric-label">Accuracy</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">9.11%</div>
        <div class="metric-label">EER</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">90.89%</div>
        <div class="metric-label">Macro F1</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">0.931</div>
        <div class="metric-label">ROC-AUC</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    MARS Open Projects 2026 &nbsp;·&nbsp; Problem Statement 2 &nbsp;·&nbsp;
    Deepfake Audio Detection &nbsp;·&nbsp; Arun Kaarthikeyan R
</div>
""", unsafe_allow_html=True)
