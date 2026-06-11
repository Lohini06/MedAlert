"""
explainer.py — SHAP explainability for MedAlert.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger


class TabularExplainer:
    """
    SHAP explainer for the XGBoost tabular model.
    """

    def __init__(self, tabular_model, feature_names: list[str]):
        self.model = tabular_model.model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(self.model)
        logger.info("TabularExplainer (TreeExplainer) ready.")

    def explain(self, X: np.ndarray) -> shap.Explanation:
        """Compute SHAP values for input array X."""
        shap_values = self.explainer(X)
        return shap_values

    def explain_single(self, x: np.ndarray) -> dict:
        """
        Explain a single sample.
        Returns dict of {feature_name: shap_value}.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        shap_values = self.explainer(x)
        values = shap_values.values[0]
        if values.ndim == 2:
            values = values[:, 1]  # positive class
        return dict(zip(self.feature_names, values))

    def plot_summary(
        self,
        X: np.ndarray,
        save_path: Path = None,
        max_display: int = 15,
    ):
        """Beeswarm summary plot."""
        shap_values = self.explainer(X)
        shap.summary_plot(
            shap_values,
            features=X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=save_path is None,
        )
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close()
            logger.info(f"Summary plot saved to {save_path}")

    def plot_waterfall(
        self,
        x: np.ndarray,
        save_path: Path = None,
    ):
        """Waterfall plot for a single prediction."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        shap_values = self.explainer(x)
        shap.waterfall_plot(
            shap_values[0],
            show=save_path is None,
        )
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close()
            logger.info(f"Waterfall plot saved to {save_path}")

    def top_features(self, x: np.ndarray, top_n: int = 5) -> list[dict]:
        """
        Return top N most influential features for a single sample.
        Used by Gradio app for human-readable explanations.
        """
        shap_dict = self.explain_single(x)
        sorted_feats = sorted(
            shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True
        )[:top_n]
        return [
            {
                "feature": feat,
                "shap_value": round(float(val), 4),
                "direction": "increases risk" if val > 0 else "decreases risk",
            }
            for feat, val in sorted_feats
        ]


class TextExplainer:
    """
    Token-level importance for BioBERT using attention weights.
    Simple but effective for clinical text visualization.
    """

    def __init__(self, text_encoder, tokenizer):
        self.encoder = text_encoder
        self.tokenizer = tokenizer

    def get_token_importance(
        self,
        text: str,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor",
    ) -> list[dict]:
        """
        Return tokens with their mean attention weight across heads.
        """
        import torch

        self.encoder.eval()
        with torch.no_grad():
            outputs = self.encoder.bert(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
                output_attentions=True,
            )

        # Average across all layers and heads → (seq_len,)
        attentions = torch.stack(outputs.attentions)   # (layers, batch, heads, seq, seq)
        mean_attn = attentions.mean(dim=(0, 2))[0]     # (seq, seq)
        token_scores = mean_attn[0].cpu().numpy()       # CLS row → (seq_len,)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())
        results = []
        for token, score in zip(tokens, token_scores):
            if token in ("[PAD]", "[CLS]", "[SEP]"):
                continue
            results.append({"token": token, "importance": round(float(score), 4)})

        return sorted(results, key=lambda x: x["importance"], reverse=True)