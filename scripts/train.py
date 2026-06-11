"""
train.py — Full training pipeline for MedAlert.
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from loguru import logger

from src.utils.config import model_cfg, train_cfg, PROCESSED_DIR
from src.utils.metrics import compute_metrics
from src.data.faers_loader import FAERSLoader
from src.data.preprocessor import MedAlertPreprocessor
from src.data.dataset import MedAlertDataset, collate_fn
from src.models.text_encoder import BioBERTEncoder
from src.models.tabular_model import TabularModel
from src.models.fusion_model import FusionClassifier, MedAlertModel


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def train_epoch(model, loader, optimizer, scheduler, device, criterion):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tab_proba      = batch["tab_proba"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        out = model(input_ids, attention_mask, tab_proba)
        loss = criterion(out["logits"], labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        all_preds.extend(out["proba"].detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    metrics["loss"] = total_loss / len(loader)
    return metrics


@torch.no_grad()
def eval_epoch(model, loader, device, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tab_proba      = batch["tab_proba"].to(device)
        labels         = batch["labels"].to(device)

        out = model(input_ids, attention_mask, tab_proba)
        loss = criterion(out["logits"], labels)

        total_loss += loss.item()
        all_preds.extend(out["proba"].cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    metrics["loss"] = total_loss / len(loader)
    return metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load data
    logger.info("Loading FAERS data...")
    loader = FAERSLoader()
    df = loader.load_demo_data(n_samples=500)
    logger.info(f"Loaded {len(df)} samples")

    # 2. Preprocess
    logger.info("Preprocessing...")
    preprocessor = MedAlertPreprocessor()
    texts, tab_features, labels = preprocessor.fit_transform(df)
    preprocessor.save(MODELS_DIR / "preprocessor.pkl")

    # 3. Train XGBoost
    logger.info("Training XGBoost...")
    split = int(0.8 * len(labels))
    tab_model = TabularModel(n_features=tab_features.shape[1])
    tab_model.fit(
        tab_features[:split], labels[:split],
        tab_features[split:], labels[split:],
    )
    tab_model.save(MODELS_DIR / "tabular_model.pkl")

    cv_results = tab_model.cross_validate(tab_features, labels)
    logger.info(f"XGBoost CV AUC: {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")

    # 4. Get XGBoost probabilities
    tab_proba = torch.tensor(
        tab_model.predict_proba(tab_features), dtype=torch.float32
    )

    # 5. Build dataset
    logger.info("Building dataset...")
    full_dataset = MedAlertDataset(texts, tab_features, labels)
    full_dataset.tab_proba = tab_proba

    train_size = int(0.8 * len(full_dataset))
    val_size   = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size,
        shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.batch_size,
        shuffle=False, collate_fn=collate_fn
    )

    # 6. Build fusion model
    text_encoder = BioBERTEncoder(freeze_base=True)
    fusion       = FusionClassifier()
    model        = MedAlertModel(text_encoder, fusion).to(device)

    # 7. Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=train_cfg.learning_rate,
                      weight_decay=train_cfg.weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=train_cfg.learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=train_cfg.epochs,
    )
    criterion = torch.nn.CrossEntropyLoss()

    # 8. Training loop
    best_val_auc = 0.0
    for epoch in range(1, train_cfg.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, criterion
        )
        val_metrics = eval_epoch(model, val_loader, device, criterion)

        logger.info(
            f"Epoch {epoch}/{train_cfg.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} AUC: {train_metrics['auc_roc']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} AUC: {val_metrics['auc_roc']:.4f}"
        )

        if val_metrics["auc_roc"] > best_val_auc:
            best_val_auc = val_metrics["auc_roc"]
            torch.save(model.state_dict(), MODELS_DIR / "best_model.pt")
            logger.info(f"  ✅ New best model saved (AUC: {best_val_auc:.4f})")

        if epoch == train_cfg.unfreeze_epoch:
            model.text_encoder.unfreeze()
            logger.info("BioBERT unfrozen for fine-tuning.")

    logger.info(f"Training complete. Best Val AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    main()