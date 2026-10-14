"""Prepare labeled, deduplicated review data and stratified data splits.

The raw JSONL dataset stays untouched. Generated CSV files are written under
data/processed/, which is ignored by Git.
"""
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

# When run as `python src/ml/prepare_data.py`, Python starts with src/ml on
# sys.path. Add the repository root so the `src.ml` package import also works.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.preprocess import normalize_text, rating_to_label

RAW_PATH = Path("data/raw/appliances_sample.jsonl")
OUTPUT_DIR = Path("data/processed")
SEED = 42


def main() -> None:
    df = pd.read_json(
        RAW_PATH,
        lines=True,
        convert_dates=False,
        dtype={"asin": str, "parent_asin": str, "user_id": str},
    )

    df["review_text"] = df["text"].map(normalize_text)
    df["label"] = df["rating"].map(rating_to_label)

    # Keep only binary-labeled reviews with non-empty normalized text.
    df = df[df["label"].notna() & df["review_text"].ne("")].copy()
    df["label"] = df["label"].astype(int)

    # Exclude all text groups whose ratings disagree on sentiment.
    labels_per_text = df.groupby("review_text")["label"].nunique()
    conflicting_texts = labels_per_text[labels_per_text > 1].index
    conflict_rows = int(df["review_text"].isin(conflicting_texts).sum())
    df = df[~df["review_text"].isin(conflicting_texts)].copy()

    # One example per normalized text prevents duplicate-text leakage between splits.
    df = df.drop_duplicates(subset="review_text", keep="first")
    df = df[["review_text", "label", "rating"]].reset_index(drop=True)

    # Reserve 30%, then split the holdout evenly into validation and test.
    train_df, holdout_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["label"],
    )
    validation_df, test_df = train_test_split(
        holdout_df,
        test_size=0.50,
        random_state=SEED,
        stratify=holdout_df["label"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    validation_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print(f"Random seed: {SEED}")
    print(f"Conflicting-text rows excluded: {conflict_rows}")
    print(f"Unique labeled reviews retained: {len(df)}")
    for name, split in (
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ):
        proportions = split["label"].value_counts(normalize=True).sort_index().round(4)
        print(f"\n{name}: {len(split)} rows")
        print("Label proportions (0=positive, 1=negative):")
        print(proportions.to_dict())


if __name__ == "__main__":
    main()
