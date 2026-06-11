"""
app.py — Gradio web app for MedAlert ADR prediction.
"""

import torch
import numpy as np
import pandas as pd
import gradio as gr
from pathlib import Path
from loguru import logger

from src.data.preprocessor import MedAlertPreprocessor
from src.models.text_encoder import BioBERTEncoder
from src.models.tabular_model import TabularModel
from src.models.fusion_model import FusionClassifier, MedAlertModel
from src.models.explainer import TabularExplainer

# ── Paths ──────────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")


# ── Load models once at startup ────────────────────────────────────────────
def load_models():
    logger.info("Loading models...")
    preprocessor = MedAlertPreprocessor.load(MODELS_DIR / "preprocessor.pkl")
    tab_model     = TabularModel.load(MODELS_DIR / "tabular_model.pkl")

    text_encoder = BioBERTEncoder()
    fusion       = FusionClassifier()
    model        = MedAlertModel(text_encoder, fusion)
    model.load_state_dict(torch.load(MODELS_DIR / "best_model.pt", map_location="cpu"))
    model.eval()

    explainer = TabularExplainer(
        tab_model,
        feature_names=preprocessor.tab_prep.feature_names,
    )

    logger.info("All models loaded.")
    return preprocessor, tab_model, model, explainer


try:
    preprocessor, tab_model, fusion_model, explainer = load_models()
    MODELS_LOADED = True
except Exception as e:
    logger.warning(f"Models not found — running in demo mode. ({e})")
    MODELS_LOADED = False


# ── Prediction function ────────────────────────────────────────────────────
def predict(
    drug_name, route, active_ingredient, reactions,
    age, weight, sex, dose_amount, dose_freq, duration,
):
    # Build a single-row dataframe
    row = pd.DataFrame([{
        "drugname":  drug_name,
        "route":     route,
        "prod_ai":   active_ingredient,
        "reactions": reactions,
        "age":       age,
        "wt":        weight,
        "sex":       sex,
        "dose_amt":  dose_amount,
        "dose_freq": dose_freq,
        "dur":       duration,
        "drug_seq":  1,
        "dechal":    "unknown",
        "rechal":    "unknown",
        "occr_country": "US",
    }])

    if not MODELS_LOADED:
        return _demo_response(drug_name, reactions)

    # Preprocess
    texts, tab_features, _ = preprocessor.transform(row)

    # XGBoost proba
    tab_proba = torch.tensor(
        tab_model.predict_proba(tab_features), dtype=torch.float32
    )

    # Tokenize text
    from transformers import AutoTokenizer
    from src.utils.config import model_cfg
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.biobert_model)
    enc = tokenizer(
        texts[0],
        max_length=model_cfg.max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Fusion prediction
    with torch.no_grad():
        out = fusion_model(
            enc["input_ids"],
            enc["attention_mask"],
            tab_proba.unsqueeze(0),
        )

    risk_score = float(out["proba"].item())
    risk_label = "🔴 HIGH RISK" if risk_score >= 0.5 else "🟢 LOW RISK"

    # SHAP explanation
    top_feats = explainer.top_features(tab_features[0], top_n=5)
    explanation = "\n".join([
        f"• {f['feature']}: {f['shap_value']:+.4f} ({f['direction']})"
        for f in top_feats
    ])

    result = f"""
## {risk_label}

**ADR Risk Score:** `{risk_score:.4f}`

---

### 🔍 Top Contributing Factors
{explanation}
"""
    return result


def _demo_response(drug_name, reactions):
    """Fallback when models aren't trained yet."""
    score = round(np.random.uniform(0.3, 0.8), 4)
    label = "🔴 HIGH RISK" if score >= 0.5 else "🟢 LOW RISK"
    return f"""
## {label} (Demo Mode)

**ADR Risk Score:** `{score}`

*Train the model first by running:*
**Drug:** {drug_name}
**Reactions:** {reactions}
"""


# ── Gradio UI ──────────────────────────────────────────────────────────────
with gr.Blocks(title="MedAlert — ADR Prediction", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 💊 MedAlert — Adverse Drug Reaction Predictor
    **Multimodal AI system combining BioBERT + XGBoost for ADR risk assessment.**
    * BioBERT + XGBoost Late Fusion| FDA FAERS Dataset*
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 💊 Drug Information")
            drug_name         = gr.Textbox(label="Drug Name", placeholder="e.g. Aspirin")
            active_ingredient = gr.Textbox(label="Active Ingredient", placeholder="e.g. acetylsalicylic acid")
            route             = gr.Dropdown(
                label="Route of Administration",
                choices=["oral", "intravenous", "intramuscular", "subcutaneous", "topical", "unknown"],
                value="oral",
            )
            reactions         = gr.Textbox(
                label="Reported Reactions",
                placeholder="e.g. nausea, dizziness, rash",
                lines=3,
            )

        with gr.Column():
            gr.Markdown("### 🧍 Patient Information")
            age         = gr.Slider(label="Age", minimum=0, maximum=100, value=55, step=1)
            weight      = gr.Slider(label="Weight (kg)", minimum=10, maximum=200, value=70, step=1)
            sex         = gr.Radio(label="Sex", choices=["M", "F", "unknown"], value="unknown")
            dose_amount = gr.Number(label="Dose Amount (mg)", value=100)
            dose_freq   = gr.Dropdown(
                label="Dose Frequency",
                choices=["once daily", "twice daily", "three times daily", "as needed", "unknown"],
                value="once daily",
            )
            duration    = gr.Slider(label="Duration (days)", minimum=1, maximum=365, value=30, step=1)

    predict_btn = gr.Button("🔍 Predict ADR Risk", variant="primary", size="lg")
    output      = gr.Markdown(label="Prediction Result")

    predict_btn.click(
        fn=predict,
        inputs=[
            drug_name, route, active_ingredient, reactions,
            age, weight, sex, dose_amount, dose_freq, duration,
        ],
        outputs=output,
    )

    gr.Markdown("""
    ---
    *MedAlert uses a late-fusion architecture: BioBERT encodes clinical text,
    XGBoost processes structured patient data, and a neural fusion layer
    combines both for the final ADR risk score.*
    """)


if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)