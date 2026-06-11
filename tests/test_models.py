"""
test_models.py — Unit tests for MedAlert models.
"""

import pytest
import numpy as np
import torch
import pandas as pd

from src.data.preprocessor import MedAlertPreprocessor, TextPreprocessor, TabularPreprocessor
from src.models.text_encoder import BioBERTEncoder
from src.models.tabular_model import TabularModel
from src.models.fusion_model import FusionClassifier, MedAlertModel


# ── Fixtures ───────────────────────────────────────────────────────────────
@pytest.fixture
def sample_df():
    return pd.DataFrame([{
        "drugname": "Aspirin", "route": "oral", "prod_ai": "acetylsalicylic acid",
        "reactions": "nausea dizziness", "age": 55, "wt": 70, "sex": "M",
        "dose_amt": 100, "dose_freq": "once daily", "dur": 30,
        "drug_seq": 1, "dechal": "unknown", "rechal": "unknown",
        "occr_country": "US", "serious": 1,
    }, {
        "drugname": "Ibuprofen", "route": "oral", "prod_ai": "ibuprofen",
        "reactions": "headache rash", "age": 30, "wt": 60, "sex": "F",
        "dose_amt": 200, "dose_freq": "twice daily", "dur": 14,
        "drug_seq": 2, "dechal": "unknown", "rechal": "unknown",
        "occr_country": "US", "serious": 0,
    }])


@pytest.fixture
def preprocessor(sample_df):
    prep = MedAlertPreprocessor()
    prep.fit_transform(sample_df)
    return prep


# ── Text Preprocessor ──────────────────────────────────────────────────────
def test_text_preprocessor_clean():
    tp = TextPreprocessor()
    result = tp.clean_text("Patient took QD dose IV for SOB")
    assert "once daily" in result
    assert "intravenous" in result
    assert "shortness of breath" in result


def test_text_preprocessor_empty():
    tp = TextPreprocessor()
    assert tp.clean_text("") == ""
    assert tp.clean_text(None) == ""


def test_text_build_combined(sample_df):
    tp = TextPreprocessor()
    text = tp.build_combined_text(sample_df.iloc[0])
    assert "aspirin" in text
    assert "oral" in text


# ── Tabular Preprocessor ───────────────────────────────────────────────────
def test_tabular_fit_transform(sample_df):
    tp = TabularPreprocessor()
    features = tp.fit_transform(sample_df)
    assert features.shape[0] == 2
    assert features.shape[1] == tp.n_features
    assert features.dtype == np.float32


def test_tabular_transform_unseen(sample_df):
    tp = TabularPreprocessor()
    tp.fit_transform(sample_df)
    features = tp.transform(sample_df)
    assert features.shape == (2, tp.n_features)


# ── Full Preprocessor ──────────────────────────────────────────────────────
def test_full_preprocessor(sample_df):
    prep = MedAlertPreprocessor()
    texts, tab_features, labels = prep.fit_transform(sample_df)
    assert len(texts) == 2
    assert tab_features.shape[0] == 2
    assert labels.shape[0] == 2
    assert set(labels).issubset({0, 1})


# ── Tabular Model ──────────────────────────────────────────────────────────
def test_tabular_model_fit_predict(sample_df):
    prep = MedAlertPreprocessor()
    _, tab_features, labels = prep.fit_transform(sample_df)

    model = TabularModel(n_features=tab_features.shape[1])
    model.fit(tab_features, labels)

    probas = model.predict_proba(tab_features)
    assert probas.shape == (2,)
    assert all(0 <= p <= 1 for p in probas)

    preds = model.predict(tab_features)
    assert set(preds).issubset({0, 1})


def test_tabular_feature_importance(sample_df):
    prep = MedAlertPreprocessor()
    _, tab_features, labels = prep.fit_transform(sample_df)

    model = TabularModel()
    model.fit(tab_features, labels)
    importance = model.get_feature_importance(prep.tab_prep.feature_names)
    assert isinstance(importance, dict)
    assert len(importance) > 0


# ── Fusion Model ───────────────────────────────────────────────────────────
def test_fusion_classifier_forward():
    model = FusionClassifier(text_dim=128, tab_dim=1, hidden_dim=64)
    text_emb = torch.randn(4, 128)
    tab_proba = torch.rand(4, 1)

    out = model(text_emb, tab_proba)
    assert "logits" in out
    assert "proba" in out
    assert out["logits"].shape == (4, 2)
    assert out["proba"].shape == (4,)
    assert all(0 <= p <= 1 for p in out["proba"].tolist())


def test_fusion_proba_scalar_input():
    """tab_proba can be 1D (batch,) — model should handle it."""
    model = FusionClassifier(text_dim=128, tab_dim=1, hidden_dim=64)
    text_emb  = torch.randn(4, 128)
    tab_proba = torch.rand(4)   # 1D

    out = model(text_emb, tab_proba)
    assert out["proba"].shape == (4,)


# ── Save / Load ────────────────────────────────────────────────────────────
def test_tabular_model_save_load(tmp_path, sample_df):
    prep = MedAlertPreprocessor()
    _, tab_features, labels = prep.fit_transform(sample_df)

    model = TabularModel()
    model.fit(tab_features, labels)

    save_path = tmp_path / "tabular_model.pkl"
    model.save(save_path)

    loaded = TabularModel.load(save_path)
    orig_probas   = model.predict_proba(tab_features)
    loaded_probas = loaded.predict_proba(tab_features)
    np.testing.assert_array_almost_equal(orig_probas, loaded_probas)


def test_preprocessor_save_load(tmp_path, sample_df):
    prep = MedAlertPreprocessor()
    prep.fit_transform(sample_df)

    save_path = tmp_path / "preprocessor.pkl"
    prep.save(save_path)

    loaded = MedAlertPreprocessor.load(save_path)
    assert loaded.tab_prep.n_features == prep.tab_prep.n_features