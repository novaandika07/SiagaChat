"""
Prepare the SiagaChat training data.

Input
    data/malicious_phish.csv      (Kaggle "Malicious URLs dataset")
    data/tranco.csv               (OPTIONAL but strongly recommended)

Output
    data/train.csv, data/test.csv   (FEATURE_NAMES + label)

What this script does
    1. drops 'defacement' (not relevant to scam links)
    2. removes duplicate URLs (ignoring scheme and 'www.')
    3. extracts host-level features (see features.py for why)
    4. adds popular benign domains from Tranco
    5. splits train/test BY DOMAIN, so the same website never appears on both
       sides (otherwise the accuracy looks better than it really is)

Run from the project root:
    python -m src.prepare_data
    python -m src.prepare_data --tranco-top 50000
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.features import FEATURE_NAMES, extract_features_and_group

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "malicious_phish.csv"
TRANCO_PATH = BASE_DIR / "data" / "tranco.csv"
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# 'defacement' is intentionally absent: those rows are dropped.
LABEL_MAPPING = {"benign": 0, "phishing": 1, "malware": 1}


def load_raw() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_PATH}\n"
            "Put malicious_phish.csv inside the data/ folder."
        )

    df = pd.read_csv(RAW_PATH, usecols=["url", "type"])
    print(f"Raw rows            : {len(df):,}")

    df["type"] = df["type"].astype(str).str.strip().str.lower()
    unknown = sorted(set(df["type"]) - set(LABEL_MAPPING) - {"defacement"})
    if unknown:
        raise ValueError(f"Unknown labels in dataset: {unknown}")

    before = len(df)
    df = df[df["type"] != "defacement"]
    print(f"Dropped defacement  : {before - len(df):,}")

    # Duplicates ignoring scheme / www / trailing slash.
    key = (
        df["url"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"^[a-z][a-z0-9+.\-]*://", "", regex=True)
        .str.replace(r"^www\.", "", regex=True)
        .str.rstrip("/")
    )
    before = len(df)
    df = df.loc[~key.duplicated()].copy()
    print(f"Dropped duplicates  : {before - len(df):,}")

    df["label"] = df["type"].map(LABEL_MAPPING).astype(int)
    df["source"] = "kaggle"
    return df[["url", "label", "source"]].reset_index(drop=True)


def load_tranco(top_n: int) -> pd.DataFrame | None:
    """Popular domains, all treated as benign (label 0)."""
    if not TRANCO_PATH.exists():
        print()
        print("!" * 62)
        print("WARNING: data/tranco.csv not found.")
        print("Without it the model sees almost no benign domain-only URLs")
        print("and tends to flag normal sites as dangerous.")
        print("Download the Tranco list (tranco-list.eu), save it as")
        print("data/tranco.csv and run this script again.")
        print("!" * 62)
        print()
        return None

    raw = pd.read_csv(TRANCO_PATH, header=None, dtype=str).dropna()
    # Tranco files are "rank,domain"; accept a plain one-column file too.
    domains = raw.iloc[:, -1].astype(str).str.strip().str.lower()
    # Skip a header row if there is one.
    domains = domains[domains != "domain"]

    ranked = domains.head(top_n)
    print(f"Tranco domains used : {len(ranked):,}")
    return pd.DataFrame(
        {"url": ranked.values, "label": 0, "source": "tranco"}
    )


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    rows, groups, labels, sources = [], [], [], []
    failures = Counter()

    print(f"Extracting {len(FEATURE_NAMES)} features from {len(df):,} URLs...")

    for i, (url, label, source) in enumerate(
        zip(df["url"], df["label"], df["source"]), start=1
    ):
        try:
            feats, group = extract_features_and_group(url)
        except Exception as exc:  # count it, never hide it
            failures[type(exc).__name__] += 1
            continue

        rows.append(feats)
        groups.append(group)
        labels.append(label)
        sources.append(source)

        if i % 50_000 == 0:
            print(f"  processed {i:,}/{len(df):,}")

    failed = sum(failures.values())
    print(f"Failed URLs skipped : {failed:,} {dict(failures) if failed else ''}")
    if failed > 0.01 * len(df):
        raise RuntimeError(
            "More than 1% of URLs failed to parse. Check features.py."
        )

    table = pd.DataFrame(rows, columns=FEATURE_NAMES)
    table["label"] = labels
    table["group"] = groups
    table["source"] = sources
    return table


def prepare(tranco_top: int) -> None:
    print("=" * 62)
    print("SiagaChat dataset preparation")
    print("=" * 62)

    frames = [load_raw()]
    tranco = load_tranco(tranco_top)
    if tranco is not None:
        frames.append(tranco)

    data = pd.concat(frames, ignore_index=True)
    table = build_feature_table(data)

    # ---- split by domain -------------------------------------------------
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(splitter.split(table, table["label"], table["group"]))

    train_df = table.iloc[train_idx].reset_index(drop=True)
    test_df = table.iloc[test_idx].reset_index(drop=True)

    overlap = set(train_df["group"]) & set(test_df["group"])
    assert not overlap, "A domain appears in both train and test"

    columns = FEATURE_NAMES + ["label"]
    train_df[columns].to_csv(TRAIN_PATH, index=False)
    test_df[columns].to_csv(TEST_PATH, index=False)

    # ---- report ----------------------------------------------------------
    print()
    print("=" * 62)
    print("Done")
    print("=" * 62)
    for name, part in (("Train", train_df), ("Test ", test_df)):
        counts = part["label"].value_counts().sort_index()
        print(
            f"{name}: {len(part):,} rows | benign {counts.get(0, 0):,} | "
            f"malicious {counts.get(1, 0):,} | "
            f"{part['group'].nunique():,} distinct domains"
        )
    print(f"Saved: {TRAIN_PATH}")
    print(f"Saved: {TEST_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tranco-top",
        type=int,
        default=50_000,
        help="how many top Tranco domains to add as benign (default 50000)",
    )
    args = parser.parse_args()
    prepare(args.tranco_top)


if __name__ == "__main__":
    main()