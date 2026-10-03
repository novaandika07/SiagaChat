"""
Training pipeline for SiagaChat.

Pipeline:
    data/train.csv
        ↓
    Train Random Forest + Logistic Regression
        ↓
    Evaluate
        ↓
    Select best model by F1
        ↓
    Save model + metadata

Binary classification:
    0 = benign
    1 = malicious
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "siagachat_url_model.joblib"
METADATA_PATH = MODEL_DIR / "siagachat_url_model_metadata.json"


# ============================================================
# FEATURE SCHEMA
# ============================================================

FEATURE_NAMES = [
    "url_length",
    "domain_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_at",
    "num_question",
    "num_equals",
    "num_ampersand",
    "num_percent",
    "num_digits",
    "num_non_alphanum",
    "num_subdomains",
    "is_ip",
    "suspicious_word_count",
    "is_shortener",
    "is_suspicious_tld",
    "domain_entropy",
    "digit_ratio",
    "is_typosquatting",
    "max_brand_similarity",
]


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(path: Path) -> pd.DataFrame:
    """Load prepared dataset."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{path}\n\n"
            "Run this first:\n"
            "python -m src.prepare_data"
        )

    df = pd.read_csv(path)

    print(f"Loaded: {path}")
    print(f"Rows: {len(df):,}")
    print()

    return df


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_dataset(df: pd.DataFrame, name: str) -> None:
    """Validate feature schema and labels."""

    expected_columns = FEATURE_NAMES + ["label"]

    actual_columns = list(df.columns)

    if actual_columns != expected_columns:

        missing = [
            col
            for col in expected_columns
            if col not in actual_columns
        ]

        unexpected = [
            col
            for col in actual_columns
            if col not in expected_columns
        ]

        raise ValueError(
            f"\n{name} has invalid columns.\n\n"
            f"Missing columns: {missing}\n"
            f"Unexpected columns: {unexpected}\n\n"
            f"Expected order:\n{expected_columns}"
        )

    if df[FEATURE_NAMES].isnull().any().any():

        null_columns = (
            df[FEATURE_NAMES]
            .columns[
                df[FEATURE_NAMES].isnull().any()
            ]
            .tolist()
        )

        raise ValueError(
            f"{name} contains missing feature values:\n"
            f"{null_columns}"
        )

    labels = set(
        pd.to_numeric(df["label"], errors="coerce")
        .dropna()
        .unique()
    )

    if not labels.issubset({0, 1}):

        raise ValueError(
            f"{name} contains invalid labels: {labels}\n"
            "Expected only 0 and 1."
        )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model_name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """Evaluate a trained model."""

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    print("=" * 60)
    print(model_name)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print()
    print("Classification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Benign",
                "Malicious",
            ],
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print()

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# ============================================================
# MAIN TRAINING
# ============================================================

def train_model():

    print("=" * 60)
    print("SiagaChat Model Training")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    print("Loading training dataset...")

    train_df = load_dataset(TRAIN_PATH)

    print("Loading testing dataset...")

    test_df = load_dataset(TEST_PATH)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    print("Validating datasets...")

    validate_dataset(
        train_df,
        "Training dataset",
    )

    validate_dataset(
        test_df,
        "Testing dataset",
    )

    print("Dataset validation passed.")
    print()

    # --------------------------------------------------------
    # Split features / labels
    # --------------------------------------------------------

    X_train = train_df[FEATURE_NAMES]
    y_train = train_df["label"].astype(int)

    X_test = test_df[FEATURE_NAMES]
    y_test = test_df["label"].astype(int)

    print("Training label distribution:")
    print(
        y_train.value_counts()
        .sort_index()
        .rename(
            index={
                0: "Benign",
                1: "Malicious",
            }
        )
    )

    print()

    print("Testing label distribution:")
    print(
        y_test.value_counts()
        .sort_index()
        .rename(
            index={
                0: "Benign",
                1: "Malicious",
            }
        )
    )

    print()

    # ========================================================
    # MODEL 1: RANDOM FOREST
    # ========================================================

    print("=" * 60)
    print("Training Random Forest")
    print("=" * 60)

    print()
    print("Configuration:")
    print("  n_estimators    = 100")
    print("  max_depth       = 20")
    print("  min_samples_leaf= 2")
    print("  class_weight    = balanced")
    print()

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    rf_model.fit(
        X_train,
        y_train,
    )

    rf_metrics = evaluate_model(
        "Random Forest",
        rf_model,
        X_test,
        y_test,
    )

    # ========================================================
    # MODEL 2: LOGISTIC REGRESSION
    # ========================================================

    print("=" * 60)
    print("Training Logistic Regression")
    print("=" * 60)

    print()
    print("Scaling features...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    lr_model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    lr_model.fit(
        X_train_scaled,
        y_train,
    )

    lr_metrics = evaluate_model(
        "Logistic Regression",
        lr_model,
        X_test_scaled,
        y_test,
    )

    # ========================================================
    # SELECT BEST MODEL
    # ========================================================

    print("=" * 60)
    print("Model Comparison")
    print("=" * 60)

    print(
        f"Random Forest F1        : "
        f"{rf_metrics['f1']:.4f}"
    )

    print(
        f"Logistic Regression F1  : "
        f"{lr_metrics['f1']:.4f}"
    )

    print()

    # RF wins ties because it generally captures
    # non-linear URL patterns better.
    if rf_metrics["f1"] >= lr_metrics["f1"]:

        selected_model = rf_model
        selected_model_type = "random_forest"
        selected_metrics = rf_metrics
        selected_scaler = None

        print("Selected model: Random Forest")

    else:

        selected_model = lr_model
        selected_model_type = "logistic_regression"
        selected_metrics = lr_metrics
        selected_scaler = scaler

        print("Selected model: Logistic Regression")

    print()

    # ========================================================
    # VALIDATE MODEL CLASSES
    # ========================================================

    if not hasattr(
        selected_model,
        "classes_",
    ):

        raise ValueError(
            "Selected model does not expose classes_."
        )

    model_classes = [
        int(value)
        for value in selected_model.classes_
    ]

    if model_classes != [0, 1]:

        raise ValueError(
            f"Unexpected model classes: "
            f"{model_classes}\n"
            "Expected [0, 1]."
        )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # For Logistic Regression we save:
    #
    # {
    #     "model": model,
    #     "scaler": scaler
    # }
    #
    # For Random Forest:
    #
    # {
    #     "model": model,
    #     "scaler": None
    # }
    # --------------------------------------------------------

    model_package = {
        "model": selected_model,
        "scaler": selected_scaler,
    }

    print("=" * 60)
    print("Saving model")
    print("=" * 60)

    joblib.dump(
        model_package,
        MODEL_PATH,
        compress=3,
    )

    print(
        f"Model saved to:\n"
        f"{MODEL_PATH}"
    )

    print()

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata = {
        "model_type": selected_model_type,
        "feature_names": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "classes": model_classes,
        "label_mapping": {
            "0": "benign",
            "1": "malicious",
        },
        "random_state": RANDOM_STATE,
        "metrics": selected_metrics,
        "dataset": {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        },
        "random_forest_config": {
            "n_estimators": 100,
            "max_depth": 20,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
        },
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print(
        f"Metadata saved to:\n"
        f"{METADATA_PATH}"
    )

    print()

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    if selected_model_type == "random_forest":

        print("=" * 60)
        print("Top Feature Importances")
        print("=" * 60)

        importances = (
            selected_model
            .feature_importances_
        )

        importance_df = pd.DataFrame(
            {
                "feature": FEATURE_NAMES,
                "importance": importances,
            }
        ).sort_values(
            "importance",
            ascending=False,
        )

        for _, row in importance_df.head(10).iterrows():

            print(
                f"{row['feature']:<30} "
                f"{row['importance']:.4f}"
            )

        print()

    # ========================================================
    # DONE
    # ========================================================

    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Selected model : "
        f"{selected_model_type}"
    )

    print(
        f"F1 Score       : "
        f"{selected_metrics['f1']:.4f}"
    )

    print(
        f"Model file     : "
        f"{MODEL_PATH}"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train_model()