"""
config.py — Central configuration for MedAlert
All hyperparameters, paths, and settings in one place.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models_saved"
LOG_DIR = ROOT_DIR / "logs"

for d in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class TextConfig:
    """BioBERT text encoder configuration."""
    model_name: str = "dmis-lab/biobert-v1.1"
    max_length: int = 256
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    dropout: float = 0.3
    hidden_dim: int = 768
    freeze_layers: int = 8


@dataclass
class TabularConfig:
    """XGBoost tabular model configuration."""
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    embedding_dim: int = 128


@dataclass
class FusionConfig:
    """Late fusion MLP configuration."""
    input_dim: int = 768 + 128
    hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    dropout: float = 0.4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_classes: int = 2


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    batch_size: int = 32
    num_epochs: int = 10
    patience: int = 3
    grad_clip: float = 1.0
    seed: int = 42
    val_split: float = 0.15
    test_split: float = 0.15
    num_workers: int = 4
    device: str = "cuda"


@dataclass
class DataConfig:
    """FAERS data configuration."""
    years: list = field(default_factory=lambda: ["2022q1", "2022q2", "2022q3", "2022q4",
                                                   "2023q1", "2023q2", "2023q3"])
    text_columns: list = field(default_factory=lambda: ["drugname", "route", "prod_ai", "outc_cod"])
    tabular_columns: list = field(default_factory=lambda: [
        "age", "sex", "wt", "occr_country", "rpsr_cod",
        "drug_seq", "dsg_drug_seq", "dose_amt", "dose_unit",
        "dose_freq", "dur", "dur_cod", "dechal", "rechal"
    ])
    target_column: str = "serious"
    max_text_samples: int = 500_000


# ── Global config instance ─────────────────────────────────────────────────
text_cfg = TextConfig()
tabular_cfg = TabularConfig()
fusion_cfg = FusionConfig()
training_cfg = TrainingConfig()
data_cfg = DataConfig()
# ── Aliases for compatibility ──────────────────────────────────────────────
@dataclass
class ModelConfig:
    biobert_model: str = "dmis-lab/biobert-v1.1"
    max_seq_length: int = 256
    text_embedding_dim: int = 128
    fusion_hidden_dim: int = 256
    dropout: float = 0.3
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05

@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    unfreeze_epoch: int = 3

model_cfg = ModelConfig()
train_cfg = TrainConfig()
MODELS_DIR = MODEL_DIR