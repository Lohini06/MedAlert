"""
fusion_model.py — Late-fusion model combining BioBERT + XGBoost for MedAlert.
"""

import torch
import torch.nn as nn
from loguru import logger

from src.utils.config import model_cfg


class FusionClassifier(nn.Module):
    """
    Late-fusion classifier that combines:
      - BioBERT text embedding  (batch, text_dim)
      - XGBoost tabular proba   (batch, 1)
    into a final ADR risk score.
    """

    def __init__(
        self,
        text_dim: int = model_cfg.text_embedding_dim,
        tab_dim: int = 1,
        hidden_dim: int = model_cfg.fusion_hidden_dim,
        dropout: float = model_cfg.dropout,
        num_classes: int = 2,
    ):
        super().__init__()

        input_dim = text_dim + tab_dim

        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self._init_weights()
        logger.info(
            f"FusionClassifier ready — "
            f"input: {input_dim}, hidden: {hidden_dim}, classes: {num_classes}"
        )

    def _init_weights(self):
        for module in self.fusion:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        text_embedding: torch.Tensor,
        tab_proba: torch.Tensor,
    ) -> dict:
        """
        Args:
            text_embedding: (batch, text_dim)  — from BioBERTEncoder
            tab_proba:      (batch, 1)          — from TabularModel.predict_proba()

        Returns:
            dict with logits (batch, 2) and proba (batch,)
        """
        if tab_proba.dim() == 1:
            tab_proba = tab_proba.unsqueeze(1)

        fused = torch.cat([text_embedding, tab_proba], dim=1)
        logits = self.fusion(fused)
        proba = torch.softmax(logits, dim=1)[:, 1]

        return {"logits": logits, "proba": proba}


class MedAlertModel(nn.Module):
    """
    Full end-to-end model:
    BioBERTEncoder → FusionClassifier
    (TabularModel runs separately as XGBoost, proba passed in)
    """

    def __init__(self, text_encoder, fusion_classifier):
        super().__init__()
        self.text_encoder = text_encoder
        self.fusion = fusion_classifier

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tab_proba: torch.Tensor,
        token_type_ids: torch.Tensor = None,
    ) -> dict:
        text_emb = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return self.fusion(text_emb, tab_proba)