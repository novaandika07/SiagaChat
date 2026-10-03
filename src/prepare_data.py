"""
Prepare SiagaChat dataset.

Input:
    data/malicious_phish.csv

Output:
    data/train.csv
    data/test.csv

Labels:
    benign   -> 0
    phishing -> 1
    malware  -> 1
    "defacement": 1,
"""

from __future__ import annotations

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features import (
    FEATURE_NAMES,
    extract_features,
    get_feature_extraction_error_types,
    get_feature_extraction_failures,
    reset_feature_extraction_failures,
)


RANDOM_STATE = 42
TEST_SIZE = 0.20

LABEL_MAPPING = {
    "benign": 0,
    "phishing": 1,
    "malware": 1,
    "defacement": 1,
}


def prepare_dataset():

    base_dir = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    input_file = os.path.join(
        base_dir,
        "data",
        "malicious_phish.csv",
    )

    train_file = os.path.join(
        base_dir,
        "data",
        "train.csv",
    )

    test_file = os.path.join(
        base_dir,
        "data",
        "test.csv",
    )

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Dataset not found: {input_file}"
        )

    print("=" * 60)
    print("SiagaChat Dataset Preparation")
    print("=" * 60)

    # --------------------------------------------------------
    # Load raw dataset
    # --------------------------------------------------------

    df = pd.read_csv(input_file)

    required_columns = {"url", "type"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing)}"
        )

    print(
        f"Raw dataset: {len(df):,} rows"
    )

    # --------------------------------------------------------
    # Normalize labels
    # --------------------------------------------------------

    df["type"] = (
        df["type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    unknown_types = sorted(
        set(df["type"].unique())
        - set(LABEL_MAPPING.keys())
    )

    if unknown_types:
        raise ValueError(
            "Unknown labels found in dataset: "
            f"{unknown_types}"
        )

    df["label"] = df["type"].map(
        LABEL_MAPPING
    )

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    reset_feature_extraction_failures()

    print(
        f"Extracting {len(FEATURE_NAMES)} features..."
    )

    feature_rows = []

    for index, url in enumerate(
        df["url"],
        start=1,
    ):

        feature_rows.append(
            extract_features(url)
        )

        if index % 10000 == 0:
            print(
                f"Processed: {index:,}/{len(df):,}"
            )

    features_df = pd.DataFrame(
        feature_rows,
        columns=FEATURE_NAMES,
    )

    # --------------------------------------------------------
    # Validate feature schema
    # --------------------------------------------------------

    if list(features_df.columns) != FEATURE_NAMES:
        raise RuntimeError(
            "Feature schema mismatch."
        )

    # --------------------------------------------------------
    # Failure report
    # --------------------------------------------------------

    failures = get_feature_extraction_failures()
    error_types = get_feature_extraction_error_types()

    print()
    print("Feature extraction failures:", failures)

    if error_types:
        print(
            "Failure types:",
            error_types,
        )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    final_df = pd.concat(
        [
            features_df,
            df[["label"]],
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # Validate labels
    # --------------------------------------------------------

    if final_df["label"].isna().any():
        raise RuntimeError(
            "Some rows have missing labels."
        )

    if not set(
        final_df["label"].unique()
    ).issubset({0, 1}):

        raise RuntimeError(
            "Labels must only contain 0 and 1."
        )

    # --------------------------------------------------------
    # Train/test split
    #
    # Stratify preserves class distribution.
    # --------------------------------------------------------

    train_df, test_df = train_test_split(
        final_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=final_df["label"],
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    train_df.to_csv(
        train_file,
        index=False,
    )

    test_df.to_csv(
        test_file,
        index=False,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Dataset preparation completed")
    print("=" * 60)

    print(
        f"Features: {len(FEATURE_NAMES)}"
    )

    print(
        f"Train: {len(train_df):,} rows"
    )

    print(
        f"Test : {len(test_df):,} rows"
    )

    print()
    print("Train label distribution:")
    print(
        train_df["label"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Test label distribution:")
    print(
        test_df["label"]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        f"Saved: {train_file}"
    )

    print(
        f"Saved: {test_file}"
    )


if __name__ == "__main__":
    prepare_dataset()