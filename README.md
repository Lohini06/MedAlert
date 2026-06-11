# 💊 MedAlert — Multimodal Adverse Drug Reaction Prediction

A multimodal AI system that predicts adverse drug reaction (ADR) risk by combining **BioBERT** (clinical text) and **XGBoost** (structured patient data) using a **late-fusion architecture** with **SHAP explainability**.

---

## 🏗️ Architecture
Clinical Text ──→ BioBERT Encoder ──→ Text Embedding (128d)
│
▼
Late Fusion MLP ──→ ADR Risk Score
▲
Structured Data ──→ XGBoost ──→ Risk Probability (1d)

**Three-stage pipeline:**
1. **BioBERT** encodes drug name, route, active ingredient, and reactions into a 128-dim embedding
2. **XGBoost** processes structured patient features (age, weight, dose, duration) into a risk probability
3. **Fusion MLP** combines both into a final ADR risk score with SHAP explanations

---

## 📁 Project Structure
MedAlert/
├── src/
│   ├── data/
│   │   ├── faers_loader.py      # FDA FAERS downloader + demo dataset
│   │   ├── preprocessor.py      # Text + tabular preprocessing
│   │   └── dataset.py           # PyTorch Dataset
│   ├── models/
│   │   ├── text_encoder.py      # BioBERT encoder
│   │   ├── tabular_model.py     # XGBoost classifier
│   │   ├── fusion_model.py      # Late-fusion MLP
│   │   └── explainer.py         # SHAP explainability
│   └── utils/
│       ├── config.py            # Hyperparameters and paths
│       └── metrics.py           # AUC, F1, precision, recall
├── scripts/
│   └── train.py                 # Full training pipeline
├── app/
│   └── app.py                   # Gradio web app
└── tests/
└── test_models.py           # Unit tests (12/12 passing)
---

## 🚀 Setup

### 1. Clone the repo
```bash
git clone https://github.com/Lohini06/MedAlert.git
cd MedAlert
```

### 2. Install dependencies
```bash
pip install -e .
```

### 3. Train the model
```bash
python scripts/train.py
```

### 4. Launch the app
```bash
python app/app.py
```

Open `http://127.0.0.1:7860` in your browser.

---

## 📊 Dataset

Uses **FDA FAERS** (Adverse Event Reporting System) — a public database of adverse drug event reports.

- Demo mode: 500 synthetic samples mirroring FAERS schema
- Real data: downloadable via `FAERSLoader.download()`

---

## 🧪 Tests

```bash
python -m pytest tests/test_models.py -v
```

All 12 tests passing across preprocessor, tabular model, fusion classifier, and save/load.

---

## 🔍 Explainability

SHAP TreeExplainer on XGBoost provides per-prediction feature importance, shown in the Gradio app as top contributing risk factors.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Clinical Text | BioBERT (dmis-lab/biobert-v1.1) |
| Structured Data | XGBoost |
| Fusion Layer | PyTorch MLP |
| Explainability | SHAP |
| Web App | Gradio |
| Data | FDA FAERS |