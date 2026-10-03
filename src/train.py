"""
Train the SiagaChat URL classifier.

    python -m src.prepare_data     (first)
    python -m src.train

Trains a Random Forest and a Logistic Regression on data/train.csv, evaluates
both on data/test.csv (a held-out set of DIFFERENT domains), keeps the one with
the better F1 and saves:

    models/siagachat_url_model.joblib
    models/siagachat_url_model_metadata.json
    reports/metrics.json
    reports/feature_importance.csv   (Random Forest only)

All numbers come from running this script. Nothing is hard-coded.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_NAMES

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
MODEL_PATH = MODEL_DIR / "siagachat_url_model.joblib"
METADATA_PATH = MODEL_DIR / "siagachat_url_model_metadata.json"

RANDOM_STATE = 42
MAX_MODEL_MB = 80  # GitHub rejects files above 100 MB

RF_PARAMS = dict(
    n_estimators=150,
    max_depth=18,
    min_samples_leaf=5,
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run first: python -m src.prepare_data"
        )
    df = pd.read_csv(path)
    expected = FEATURE_NAMES + ["label"]
    if list(df.columns) != expected:
        raise ValueError(
            f"{path.name} columns do not match features.py.\n"
            "features.py changed after prepare_data? Run prepare_data again."
        )
    if df[FEATURE_NAMES].isnull().any().any():
        raise ValueError(f"{path.name} has missing feature values.")
    return df


def evaluate(name: str, y_true, y_pred) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["confusion"] = {
        "true_benign": int(tn),
        "false_alarm": int(fp),
        "missed_scam": int(fn),
        "caught_scam": int(tp),
    }

    print("=" * 62)
    print(name)
    print("=" * 62)
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}  (of links flagged, how many were scams)")
    print(f"Recall    : {metrics['recall']:.4f}  (of all scams, how many were caught)")
    print(f"F1        : {metrics['f1']:.4f}")
    print(f"Safe links wrongly flagged : {fp:,}")
    print(f"Scam links missed          : {fn:,}")
    print()
    return metrics


def train() -> None:
    print("=" * 62)
    print("SiagaChat model training")
    print("=" * 62)

    train_df = load(TRAIN_PATH)
    test_df = load(TEST_PATH)

    X_train, y_train = train_df[FEATURE_NAMES], train_df["label"].astype(int)
    X_test, y_test = test_df[FEATURE_NAMES], test_df["label"].astype(int)

    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")
    print(f"Malicious share  train {y_train.mean():.1%} | test {y_test.mean():.1%}")
    print()

    # ---- Random Forest ---------------------------------------------------
    print("Training Random Forest...")
    rf = RandomForestClassifier(**RF_PARAMS).fit(X_train, y_train)
    rf_metrics = evaluate("Random Forest", y_test, rf.predict(X_test))

    # ---- Logistic Regression (baseline) ----------------------------------
    print("Training Logistic Regression...")
    scaler = StandardScaler().fit(X_train)
    lr = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
    ).fit(scaler.transform(X_train), y_train)
    lr_metrics = evaluate(
        "Logistic Regression (baseline)", y_test, lr.predict(scaler.transform(X_test))
    )

    # ---- choose ----------------------------------------------------------
    if rf_metrics["f1"] >= lr_metrics["f1"]:
        model, model_type, metrics, used_scaler = rf, "random_forest", rf_metrics, None
    else:
        model, model_type, metrics, used_scaler = (
            lr,
            "logistic_regression",
            lr_metrics,
            scaler,
        )
    print(f"Selected model: {model_type}\n")

    classes = [int(c) for c in model.classes_]
    if classes != [0, 1]:
        raise ValueError(f"Unexpected classes {classes}, expected [0, 1]")

    # ---- save ------------------------------------------------------------
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    joblib.dump({"model": model, "scaler": used_scaler}, MODEL_PATH, compress=3)
    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"Model saved: {MODEL_PATH}  ({size_mb:.1f} MB)")
    if size_mb > MAX_MODEL_MB:
        print(
            f"WARNING: model is bigger than {MAX_MODEL_MB} MB. Lower "
            "n_estimators / max_depth in RF_PARAMS, or GitHub will reject it."
        )

    metadata = {
        "model_type": model_type,
        "feature_names": FEATURE_NAMES,
        "classes": classes,
        "label_mapping": {"0": "benign", "1": "malicious"},
        "metrics": metrics,
        "split": "grouped by registered domain (no domain in both train and test)",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(
            {"random_forest": rf_metrics, "logistic_regression": lr_metrics,
             "selected": model_type},
            indent=2,
        ),
        encoding="utf-8",
    )

    if model_type == "random_forest":
        importance = (
            pd.DataFrame(
                {"feature": FEATURE_NAMES, "importance": model.feature_importances_}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        importance.to_csv(REPORT_DIR / "feature_importance.csv", index=False)
        print("\nTop features:")
        for _, row in importance.head(8).iterrows():
            print(f"  {row['feature']:<24} {row['importance']:.3f}")

    print("\nNext: python -m src.sanity_check")


if __name__ == "__main__":
    train()