"""
preprocessor.py — Text + tabular preprocessing pipeline for MedAlert.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from loguru import logger

from src.utils.config import PROCESSED_DIR, data_cfg


class TextPreprocessor:
    """Clean and prepare clinical text for BioBERT."""

    MEDICAL_ABBREVIATIONS = {
        "QD": "once daily", "BID": "twice daily", "TID": "three times daily",
        "QID": "four times daily", "PRN": "as needed", "IV": "intravenous",
        "IM": "intramuscular", "SC": "subcutaneous", "PO": "oral",
        "SOB": "shortness of breath", "CP": "chest pain", "HA": "headache",
    }

    def clean_text(self, text: str) -> str:
        """Clean a single text string."""
        if not isinstance(text, str) or not text.strip():
            return ""
        for abbr, expansion in self.MEDICAL_ABBREVIATIONS.items():
            text = re.sub(rf"\b{abbr}\b", expansion, text, flags=re.IGNORECASE)
        text = re.sub(r"[^\w\s\-\.\,\(\)]", " ", text)
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text

    def build_combined_text(self, row: pd.Series) -> str:
        """Combine multiple text columns into a single clinical narrative."""
        parts = []
        if pd.notna(row.get("drugname")):
            parts.append(f"Drug: {row['drugname']}")
        if pd.notna(row.get("route")):
            parts.append(f"Route: {row['route']}")
        if pd.notna(row.get("prod_ai")):
            parts.append(f"Active ingredient: {row['prod_ai']}")
        if pd.notna(row.get("reactions")):
            parts.append(f"Reactions: {row['reactions']}")
        return self.clean_text(". ".join(parts))

    def process_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """Apply text processing to full dataframe."""
        logger.info("Processing clinical text...")
        texts = df.apply(self.build_combined_text, axis=1)
        texts = texts.replace("", "no clinical information available")
        logger.info(f"Avg text length: {texts.str.split().str.len().mean():.1f} words")
        return texts


class TabularPreprocessor:
    """Preprocess structured patient features."""

    CATEGORICAL_COLS = ["sex", "occr_country", "dose_freq", "dechal", "rechal", "route"]
    NUMERICAL_COLS = ["age", "wt", "drug_seq", "dose_amt", "dur"]

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.feature_names = []
        self._fitted = False

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        cat_df = pd.DataFrame(index=df.index)
        for col in self.CATEGORICAL_COLS:
            if col not in df.columns:
                cat_df[col] = 0
                continue
            series = df[col].astype(str).fillna("UNKNOWN")
            if fit:
                le = LabelEncoder()
                cat_df[col] = le.fit_transform(series)
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    known = set(le.classes_)
                    series = series.apply(lambda x: x if x in known else "UNKNOWN")
                    if "UNKNOWN" not in known:
                        series = series.apply(lambda x: le.classes_[0])
                    cat_df[col] = le.transform(series)
                else:
                    cat_df[col] = 0
        return cat_df

    def _process_numericals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        num_df = pd.DataFrame(index=df.index)
        for col in self.NUMERICAL_COLS:
            if col in df.columns:
                num_df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                num_df[col] = np.nan

        if fit:
            num_arr = self.num_imputer.fit_transform(num_df)
            num_arr = self.scaler.fit_transform(num_arr)
        else:
            num_arr = self.num_imputer.transform(num_df)
            num_arr = self.scaler.transform(num_arr)

        return pd.DataFrame(num_arr, columns=self.NUMERICAL_COLS, index=df.index)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific feature interactions."""
        feat = pd.DataFrame(index=df.index)
        age = pd.to_numeric(df.get("age", pd.Series(dtype=float)), errors="coerce").fillna(55)
        feat["is_elderly"] = (age > 65).astype(int)
        feat["is_pediatric"] = (age < 18).astype(int)
        feat["polypharmacy"] = (pd.to_numeric(df.get("drug_seq", 1),
                                               errors="coerce").fillna(1) > 5).astype(int)
        feat["high_dose"] = (pd.to_numeric(df.get("dose_amt", 0),
                                            errors="coerce").fillna(0) > 500).astype(int)
        feat["long_duration"] = (pd.to_numeric(df.get("dur", 0),
                                                errors="coerce").fillna(0) > 90).astype(int)
        return feat

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        cat_df = self._encode_categoricals(df, fit=True)
        num_df = self._process_numericals(df, fit=True)
        eng_df = self._engineer_features(df)
        combined = pd.concat([num_df, cat_df, eng_df], axis=1)
        self.feature_names = list(combined.columns)
        self._fitted = True
        logger.info(f"Tabular features: {len(self.feature_names)} columns")
        return combined.values.astype(np.float32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit_transform() first.")
        cat_df = self._encode_categoricals(df, fit=False)
        num_df = self._process_numericals(df, fit=False)
        eng_df = self._engineer_features(df)
        combined = pd.concat([num_df, cat_df, eng_df], axis=1)
        for col in self.feature_names:
            if col not in combined.columns:
                combined[col] = 0
        return combined[self.feature_names].values.astype(np.float32)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


class MedAlertPreprocessor:
    """Master preprocessor combining text + tabular pipelines."""

    def __init__(self):
        self.text_prep = TextPreprocessor()
        self.tab_prep = TabularPreprocessor()

    def fit_transform(self, df: pd.DataFrame):
        texts = self.text_prep.process_dataframe(df)
        tab_features = self.tab_prep.fit_transform(df)
        labels = df["serious"].values.astype(np.int64) if "serious" in df.columns else None
        return texts.tolist(), tab_features, labels

    def transform(self, df: pd.DataFrame):
        texts = self.text_prep.process_dataframe(df)
        tab_features = self.tab_prep.transform(df)
        labels = df["serious"].values.astype(np.int64) if "serious" in df.columns else None
        return texts.tolist(), tab_features, labels

    def save(self, path: Path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor saved to {path}")

    @staticmethod
    def load(path: Path):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)