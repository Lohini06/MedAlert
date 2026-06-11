"""
dataset.py — PyTorch Dataset for MedAlert multimodal training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from loguru import logger

from src.utils.config import model_cfg


class MedAlertDataset(Dataset):
    def __init__(
        self,
        texts: list,
        tab_features: np.ndarray,
        labels: np.ndarray = None,
        tokenizer_name: str = model_cfg.biobert_model,
        max_length: int = model_cfg.max_seq_length,
    ):
        self.texts = texts
        self.tab_features = torch.tensor(tab_features, dtype=torch.float32)
        self.labels = (
            torch.tensor(labels, dtype=torch.long) if labels is not None else None
        )
        self.tab_proba = None
        self.max_length = max_length

        logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        logger.info(f"Dataset ready — {len(self.texts)} samples, "
                    f"{tab_features.shape[1]} tabular features")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "tab_features":   self.tab_features[idx],
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        if self.tab_proba is not None:
            item["tab_proba"] = self.tab_proba[idx]

        if self.labels is not None:
            item["labels"] = self.labels[idx]

        return item


class InferenceDataset(Dataset):
    def __init__(
        self,
        texts: list,
        tab_features: np.ndarray,
        tokenizer_name: str = model_cfg.biobert_model,
        max_length: int = model_cfg.max_seq_length,
    ):
        self.texts = texts
        self.tab_features = torch.tensor(tab_features, dtype=torch.float32)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "tab_features":   self.tab_features[idx],
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
        return item


def collate_fn(batch):
    keys = batch[0].keys()
    out = {}
    for key in keys:
        out[key] = torch.stack([sample[key] for sample in batch])
    return out