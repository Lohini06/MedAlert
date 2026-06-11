"""
text_encoder.py — BioBERT text encoder for MedAlert.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from loguru import logger

from src.utils.config import model_cfg


class BioBERTEncoder(nn.Module):
    """
    Wraps a pretrained BioBERT model and produces a fixed-size
    embedding from clinical text via [CLS] token pooling.
    """

    def __init__(
        self,
        model_name: str = model_cfg.biobert_model,
        output_dim: int = model_cfg.text_embedding_dim,
        dropout: float = model_cfg.dropout,
        freeze_base: bool = False,
    ):
        super().__init__()
        logger.info(f"Loading BioBERT: {model_name}")
        self.bert = AutoModel.from_pretrained(model_name)

        hidden_size = self.bert.config.hidden_size  # 768 for bert-base

        # Projection head: 768 → output_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if freeze_base:
            self._freeze_base_layers()

        logger.info(
            f"BioBERTEncoder ready — hidden: {hidden_size}, "
            f"output: {output_dim}, frozen: {freeze_base}"
        )

    def _freeze_base_layers(self):
        """Freeze all BERT weights; only projection trains."""
        for param in self.bert.parameters():
            param.requires_grad = False
        logger.info("BioBERT base layers frozen.")

    def unfreeze(self):
        """Unfreeze for fine-tuning after warm-up."""
        for param in self.bert.parameters():
            param.requires_grad = True
        logger.info("BioBERT base layers unfrozen.")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)
            token_type_ids: (batch, seq_len) — optional

        Returns:
            embedding: (batch, output_dim)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # [CLS] token — index 0 of last hidden state
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, 768)

        return self.projection(cls_embedding)  # (batch, output_dim)

    @property
    def output_dim(self) -> int:
        return self.projection[0].out_features