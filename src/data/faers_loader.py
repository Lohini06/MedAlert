"""
faers_loader.py — FDA FAERS data downloader and loader.
"""

import os
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger

from src.utils.config import RAW_DIR, PROCESSED_DIR, data_cfg


FAERS_BASE_URL = "https://fis.fda.gov/content/Exports"

FAERS_FILES = {
    "2022q1": "faers_ascii_2022Q1.zip",
    "2022q2": "faers_ascii_2022Q2.zip",
    "2022q3": "faers_ascii_2022Q3.zip",
    "2022q4": "faers_ascii_2022Q4.zip",
    "2023q1": "faers_ascii_2023Q1.zip",
    "2023q2": "faers_ascii_2023Q2.zip",
    "2023q3": "faers_ascii_2023Q3.zip",
}


def download_faers_quarter(quarter: str, dest_dir: Path = RAW_DIR) -> Path:
    filename = FAERS_FILES[quarter]
    url = f"{FAERS_BASE_URL}/{filename}"
    dest_path = dest_dir / filename

    if dest_path.exists():
        logger.info(f"Already downloaded: {filename}")
        return dest_path

    logger.info(f"Downloading {filename} from FDA FAERS...")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    logger.info(f"Downloaded: {dest_path}")
    return dest_path


def extract_faers_zip(zip_path: Path, dest_dir: Path = RAW_DIR) -> Path:
    quarter = zip_path.stem.split("_")[-1].lower()
    extract_dir = dest_dir / quarter

    if extract_dir.exists():
        logger.info(f"Already extracted: {extract_dir}")
        return extract_dir

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    logger.info(f"Extracted to: {extract_dir}")
    return extract_dir


def load_faers_quarter(extract_dir: Path) -> pd.DataFrame:
    ascii_dir = extract_dir / "ASCII"
    if not ascii_dir.exists():
        ascii_dir = extract_dir

    def read_table(pattern: str) -> pd.DataFrame:
        files = list(ascii_dir.glob(f"{pattern}*.txt"))
        if not files:
            files = list(ascii_dir.glob(f"{pattern}*.TXT"))
        if not files:
            logger.warning(f"No file found for pattern: {pattern}")
            return pd.DataFrame()
        return pd.read_csv(files[0], sep="$", encoding="latin-1",
                           on_bad_lines="skip", low_memory=False)

    demo = read_table("DEMO")
    drug = read_table("DRUG")
    reac = read_table("REAC")
    outc = read_table("OUTC")

    for df in [demo, drug, reac, outc]:
        df.columns = df.columns.str.lower().str.strip()

    if demo.empty:
        return pd.DataFrame()

    if not drug.empty and "primaryid" in drug.columns:
        drug_agg = (drug.groupby("primaryid")
                    .agg(drugname=("drugname", lambda x: " | ".join(x.dropna().unique())),
                         route=("route", lambda x: " | ".join(x.dropna().unique())),
                         prod_ai=("prod_ai", lambda x: " | ".join(x.dropna().unique())))
                    .reset_index())
        df = demo.merge(drug_agg, on="primaryid", how="left")
    else:
        df = demo.copy()

    if not reac.empty and "primaryid" in reac.columns:
        reac_agg = (reac.groupby("primaryid")["pt"]
                    .apply(lambda x: " | ".join(x.dropna().unique()))
                    .reset_index()
                    .rename(columns={"pt": "reactions"}))
        df = df.merge(reac_agg, on="primaryid", how="left")

    if not outc.empty and "primaryid" in outc.columns:
        serious_codes = {"DE", "LT", "HO", "DS", "CA", "RI"}
        outc["is_serious"] = outc["outc_cod"].isin(serious_codes).astype(int)
        outc_agg = (outc.groupby("primaryid")["is_serious"]
                    .max()
                    .reset_index()
                    .rename(columns={"is_serious": "serious"}))
        df = df.merge(outc_agg, on="primaryid", how="left")
        df["serious"] = df["serious"].fillna(0).astype(int)

    return df


def load_all_quarters(quarters: list = None, sample_size: int = None) -> pd.DataFrame:
    quarters = quarters or data_cfg.years
    all_dfs = []

    for quarter in quarters:
        zip_path = RAW_DIR / FAERS_FILES.get(quarter, "")
        if not zip_path.exists():
            logger.warning(f"File not found for {quarter}. Skipping.")
            continue

        extract_dir = extract_faers_zip(zip_path)
        df = load_faers_quarter(extract_dir)

        if not df.empty:
            df["quarter"] = quarter
            all_dfs.append(df)
            logger.info(f"Loaded {len(df):,} records from {quarter}")

    if not all_dfs:
        logger.error("No data loaded. Run download_faers_quarter() first.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["primaryid"])

    if sample_size:
        combined = combined.sample(n=min(sample_size, len(combined)), random_state=42)

    logger.info(f"Total records loaded: {len(combined):,}")
    return combined


def create_demo_dataset(n_samples: int = 10_000, save: bool = True) -> pd.DataFrame:
    np.random.seed(42)
    drugs = ["Aspirin", "Metformin", "Lisinopril", "Atorvastatin", "Amoxicillin",
             "Ibuprofen", "Omeprazole", "Amlodipine", "Metoprolol", "Warfarin"]
    reactions = ["nausea", "headache", "dizziness", "rash", "fatigue",
                 "chest pain", "dyspnea", "bleeding", "liver injury", "anaphylaxis"]
    routes = ["oral", "intravenous", "subcutaneous", "intramuscular", "topical"]

    df = pd.DataFrame({
        "primaryid": range(n_samples),
        "age": np.random.normal(55, 18, n_samples).clip(1, 100).astype(int),
        "sex": np.random.choice(["M", "F", "UNK"], n_samples, p=[0.45, 0.45, 0.10]),
        "wt": np.random.normal(75, 20, n_samples).clip(30, 200),
        "occr_country": np.random.choice(["US", "GB", "DE", "FR", "JP"], n_samples),
        "drug_seq": np.random.randint(1, 10, n_samples),
        "dose_amt": np.random.choice([10, 25, 50, 100, 200, 500], n_samples),
        "dose_freq": np.random.choice(["QD", "BID", "TID", "QID", "PRN"], n_samples),
        "dur": np.random.randint(1, 365, n_samples),
        "dechal": np.random.choice(["Y", "N", "U"], n_samples),
        "rechal": np.random.choice(["Y", "N", "U"], n_samples),
        "drugname": [np.random.choice(drugs) for _ in range(n_samples)],
        "route": np.random.choice(routes, n_samples),
        "prod_ai": [np.random.choice(drugs) for _ in range(n_samples)],
        "reactions": [" | ".join(np.random.choice(reactions,
                      np.random.randint(1, 4), replace=False)) for _ in range(n_samples)],
        "quarter": "demo",
    })

    risk = (
        (df["age"] > 65).astype(float) * 0.3 +
        (df["drug_seq"] > 5).astype(float) * 0.2 +
        df["reactions"].str.contains("bleeding|anaphylaxis|liver").astype(float) * 0.4 +
        np.random.uniform(0, 0.3, n_samples)
    )
    df["serious"] = (risk > 0.5).astype(int)

    logger.info(f"Demo dataset: {len(df):,} samples, {df['serious'].mean():.2%} serious")

    if save:
        path = PROCESSED_DIR / "demo_dataset.csv"
        df.to_csv(path, index=False)
        logger.info(f"Demo dataset saved to {path}")

    return df


class FAERSLoader:
    """Convenience wrapper used by train.py and app.py."""

    def load_demo_data(self, n_samples: int = 10_000) -> pd.DataFrame:
        return create_demo_dataset(n_samples=n_samples, save=True)

    def load_real_data(self, quarters: list = None, sample_size: int = None) -> pd.DataFrame:
        return load_all_quarters(quarters=quarters, sample_size=sample_size)

    def download(self, quarters: list = None):
        quarters = quarters or data_cfg.years
        for q in quarters:
            download_faers_quarter(q)


if __name__ == "__main__":
    df = create_demo_dataset(n_samples=1000)
    print(df.head())
    print(df["serious"].value_counts())