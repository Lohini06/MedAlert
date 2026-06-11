"""
tabular_model.py — XGBoost tabular model for MedAlert.
"""

import numpy as np
import pickle
from pathlib import Path
from loguru import logger

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.utils.config import model_cfg, PROCESSED_DIR


class TabularModel:
    """
    XGBoost classifier for structured patient features.
    Produces both predictions and probability scores for fusion.
    """

    def __init__(self, n_features: int = None):
        self.n_features = n_features
        self.model = xgb.XGBClassifier(
            n_estimators=model_cfg.xgb_n_estimators,
            max_depth=model_cfg.xgb_max_depth,
            learning_rate=model_cfg.xgb_learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
        )
        self._fitted = False
        logger.info("TabularModel (XGBoost) initialized.")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ):
        """Train XGBoost with optional validation set for early stopping."""
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        logger.info(f"Training XGBoost on {X_train.shape[0]} samples...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=50,
        )
        self._fitted = True

        train_auc = roc_auc_score(y_train, self.predict_proba(X_train))
        logger.info(f"Train AUC: {train_auc:.4f}")

        if X_val is not None:
            val_auc = roc_auc_score(y_val, self.predict_proba(X_val))
            logger.info(f"Val AUC: {val_auc:.4f}")

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
    ) -> dict:
        """Stratified K-Fold cross-validation."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        aucs = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            fold_model = xgb.XGBClassifier(
                n_estimators=model_cfg.xgb_n_estimators,
                max_depth=model_cfg.xgb_max_depth,
                learning_rate=model_cfg.xgb_learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="auc",
                random_state=42,
                n_jobs=-1,
            )
            fold_model.fit(X_tr, y_tr, verbose=False)
            preds = fold_model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, preds)
            aucs.append(auc)
            logger.info(f"Fold {fold+1}/{n_splits} — AUC: {auc:.4f}")

        results = {"mean_auc": np.mean(aucs), "std_auc": np.std(aucs), "fold_aucs": aucs}
        logger.info(f"CV AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
        return results

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of positive class. Shape: (n_samples,)"""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_feature_importance(self, feature_names: list[str] = None) -> dict:
        """Return feature importances as a sorted dict."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        scores = self.model.feature_importances_
        names = feature_names or [f"f{i}" for i in range(len(scores))]
        importance = dict(sorted(zip(names, scores), key=lambda x: x[1], reverse=True))
        return importance

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"TabularModel saved to {path}")

    @staticmethod
    def load(path: Path) -> "TabularModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"TabularModel loaded from {path}")
        return model