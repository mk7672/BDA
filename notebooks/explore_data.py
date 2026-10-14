"""Phase 1 exploration of the raw Amazon Appliances sample. Read-only: changes no data."""
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/appliances_sample.jsonl")
SEED = 42

pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", 200)


def section(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


# dtype is pinned for ID columns so digit-only IDs are not turned into integers
df = pd.read_json(
    RAW_PATH,
    lines=True,
    convert_dates=False,
    dtype={"asin": str, "parent_asin": str, "user_id": str},
)
df = df.drop(columns=["images"])  # list of image URLs, not needed for text classification

section("SHAPE AND TYPES")
print(df.shape)
print(df.dtypes)

section("MISSING VALUES PER COLUMN")
print(df.isna().sum())

section("RATING DISTRIBUTION")
counts = df["rating"].value_counts().sort_index()
print(pd.DataFrame({"count": counts, "percent": (counts / len(df) * 100).round(2)}))

section("RATING MIX BY POSITION IN FILE (is the sample ordered?)")
block = df.index * 4 // len(df)  # 0..3 = which quarter of the file
print(pd.crosstab(block, df["rating"], normalize="index").round(3))

section("TIMESTAMP RANGE")
ts = pd.to_datetime(df["timestamp"], unit="ms")
print(ts.min(), "->", ts.max())

section("EMPTY OR WHITESPACE-ONLY TEXT")
text = df["text"].fillna("")
print("text empty:", int((text.str.strip() == "").sum()))
print("title empty:", int((df["title"].fillna("").str.strip() == "").sum()))

section("DUPLICATES")
print("duplicate text:", int(df.duplicated(subset=["text"]).sum()))
print("duplicate (user, product, text):", int(df.duplicated(subset=["user_id", "parent_asin", "text"]).sum()))
print("\nMost frequent texts:")
print(text.value_counts().head(10))

section("REVIEW LENGTH IN WORDS")
words = text.str.split().str.len()
print(words.describe(percentiles=[0.5, 0.9, 0.99]).round(1))
print("2 words or fewer:", int((words <= 2).sum()))
print("over 300 words:", int((words > 300).sum()))

section("HTML OR ENTITY ARTIFACTS")
print("rows with HTML tags or entities:", int(text.str.contains(r"<[^>]+>|&[a-zA-Z]+;|&#\d+;", regex=True).sum()))

section("RANDOM SAMPLES (seed=42)")
for label, subset in [("NEGATIVE (rating <= 2)", df[df["rating"] <= 2]),
                      ("NEUTRAL (rating == 3)", df[df["rating"] == 3]),
                      ("POSITIVE (rating >= 4)", df[df["rating"] >= 4])]:
    print(f"\n--- {label} ---")
    for _, row in subset.sample(4, random_state=SEED).iterrows():
        print(f"[{row['rating']}] {str(row['title'])[:60]} | {str(row['text'])[:200]}")
