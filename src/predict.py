"""
SiagaChat URL Predictor

Prediction pipeline:

    URL
     ↓
    extract_features()
     ↓
    DataFrame dengan FEATURE_NAMES
     ↓
    Load model + scaler
     ↓
    predict_proba()
     ↓
    Risk score
     ↓
    Human-readable explanation

Model package dibuat oleh src.train:
    {
        "model": model,
        "scaler": scaler
    }

Feature schema dibaca dari metadata JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.features import (
    FEATURE_NAMES,
    extract_features,
    explain_features,
)


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "siagachat_url_model.joblib"
)

METADATA_PATH = (
    BASE_DIR
    / "models"
    / "siagachat_url_model_metadata.json"
)


# ============================================================
# THRESHOLDS
# ============================================================

SAFE_THRESHOLD = 30
WARNING_THRESHOLD = 70


# ============================================================
# LOAD MODEL
# ============================================================

def load_model_package():
    """Load trained model package."""

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model tidak ditemukan:\n"
            f"{MODEL_PATH}\n\n"
            f"Jalankan:\n"
            f"python -m src.train"
        )

    package = joblib.load(MODEL_PATH)

    # --------------------------------------------------------
    # Expected format from current train.py:
    #
    # {
    #     "model": model,
    #     "scaler": scaler
    # }
    # --------------------------------------------------------

    if not isinstance(package, dict):

        raise ValueError(
            "Format model tidak dikenali.\n"
            "Model harus dibuat menggunakan train.py terbaru."
        )

    if "model" not in package:

        raise ValueError(
            "Model package tidak memiliki key 'model'."
        )

    model = package["model"]
    scaler = package.get("scaler")

    return model, scaler


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata() -> dict:
    """Load model metadata."""

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"Metadata tidak ditemukan:\n"
            f"{METADATA_PATH}\n\n"
            f"Jalankan training ulang dengan:\n"
            f"python -m src.train"
        )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    return metadata


# ============================================================
# VALIDATE METADATA
# ============================================================

def validate_metadata(metadata: dict) -> list[str]:
    """Validate feature schema from metadata."""

    if "feature_names" not in metadata:

        raise ValueError(
            "Metadata tidak memiliki 'feature_names'.\n"
            "Gunakan metadata yang dibuat oleh train.py terbaru."
        )

    feature_names = metadata["feature_names"]

    if not isinstance(feature_names, list):

        raise ValueError(
            "'feature_names' pada metadata harus berupa list."
        )

    # --------------------------------------------------------
    # Ensure prediction schema matches features.py
    # --------------------------------------------------------

    if feature_names != FEATURE_NAMES:

        raise ValueError(
            "Feature schema tidak sinkron!\n\n"
            f"Metadata:\n{feature_names}\n\n"
            f"features.py:\n{FEATURE_NAMES}\n\n"
            "Pastikan features.py dan model berasal "
            "dari versi pipeline yang sama."
        )

    return feature_names


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input(
    url: str,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Extract features and create a pandas DataFrame.

    Using DataFrame preserves feature names and order.
    """

    features = extract_features(url)

    # --------------------------------------------------------
    # Validate all required features exist
    # --------------------------------------------------------

    missing = [
        name
        for name in feature_names
        if name not in features
    ]

    if missing:

        raise ValueError(
            "Feature extraction tidak lengkap.\n"
            f"Missing features: {missing}"
        )

    # --------------------------------------------------------
    # Ignore anything outside the official schema.
    # --------------------------------------------------------

    values = {
        name: features[name]
        for name in feature_names
    }

    # --------------------------------------------------------
    # IMPORTANT:
    # DataFrame prevents sklearn feature-name warning.
    # --------------------------------------------------------

    X = pd.DataFrame(
        [values],
        columns=feature_names,
    )

    return X


# ============================================================
# GET MALICIOUS PROBABILITY
# ============================================================

def get_malicious_probability(
    model,
    scaler,
    X: pd.DataFrame,
) -> float:
    """
    Get probability for class 1 = malicious.
    """

    # --------------------------------------------------------
    # Logistic Regression
    #
    # Needs StandardScaler.
    # --------------------------------------------------------

    if scaler is not None:

        X_input = scaler.transform(X)

    else:

        # ----------------------------------------------------
        # Random Forest
        #
        # Keep DataFrame so feature names remain available.
        # ----------------------------------------------------

        X_input = X

    # --------------------------------------------------------
    # Predict probability
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_input
    )

    # --------------------------------------------------------
    # Find class index dynamically.
    #
    # Do NOT blindly use probabilities[0][1].
    # --------------------------------------------------------

    if not hasattr(model, "classes_"):

        raise ValueError(
            "Model tidak memiliki classes_."
        )

    classes = list(model.classes_)

    if 1 not in classes:

        raise ValueError(
            f"Model tidak memiliki class 1.\n"
            f"Classes: {classes}"
        )

    malicious_index = classes.index(1)

    malicious_probability = float(
        probabilities[0][malicious_index]
    )

    return malicious_probability


# ============================================================
# RISK LABEL
# ============================================================

def risk_label(score: float) -> str:
    """Convert probability score into human-readable label."""

    if score < SAFE_THRESHOLD:

        return "Aman"

    if score < WARNING_THRESHOLD:

        return "Waspada"

    return "Bahaya"


# ============================================================
# PREDICT URL
# ============================================================

def predict_url(url: str) -> dict:
    """Analyze a URL."""

    url = str(url).strip()

    if not url:

        raise ValueError(
            "URL tidak boleh kosong."
        )

    # --------------------------------------------------------
    # Load model + metadata
    # --------------------------------------------------------

    model, scaler = load_model_package()

    metadata = load_metadata()

    feature_names = validate_metadata(
        metadata
    )

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    X = prepare_input(
        url,
        feature_names,
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    malicious_probability = (
        get_malicious_probability(
            model,
            scaler,
            X,
        )
    )

    # --------------------------------------------------------
    # Convert to 0-100
    # --------------------------------------------------------

    risk_score = round(
        malicious_probability * 100,
        2,
    )

    label = risk_label(
        risk_score
    )

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    reasons = explain_features(
        url
    )

    return {
        "url": url,
        "risk_score": risk_score,
        "label": label,
        "malicious_probability": malicious_probability,
        "reasons": reasons,
        "features": X.iloc[0].to_dict(),
        "model_type": metadata.get(
            "model_type",
            "unknown",
        ),
    }


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(result: dict) -> None:
    """Print prediction result."""

    print()
    print("=" * 60)
    print("SiagaChat Analysis Result")
    print("=" * 60)

    print()
    print(f"URL   : {result['url']}")
    print(f"Status: {result['label']}")
    print(
        f"Score : {result['risk_score']:.2f}/100"
    )

    print(
        f"Model : {result['model_type']}"
    )

    print()

    print("-" * 60)
    print("Reasons")
    print("-" * 60)

    reasons = result["reasons"]

    if reasons:

        for reason in reasons:
            print(f"  • {reason}")

    else:

        print(
            "  • Tidak ditemukan indikasi mencurigakan "
            "berdasarkan rule explanation."
        )

    print()

    print("-" * 60)
    print("ML Features")
    print("-" * 60)

    features = result["features"]

    for name, value in features.items():

        if isinstance(value, float):

            print(
                f"  {name:<25}: {value:.4f}"
            )

        else:

            print(
                f"  {name:<25}: {value}"
            )

    print()
    print("=" * 60)
    print()


# ============================================================
# INTERACTIVE MODE
# ============================================================

def main():

    print("=" * 60)
    print("SiagaChat URL Predictor")
    print("=" * 60)

    print()
    print(
        "Enter a URL to analyze "
        "(or 'quit' to exit):"
    )
    print()

    while True:

        try:

            url = input("URL: ").strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print()
            print("Exiting SiagaChat...")
            break

        if url.lower() in {
            "quit",
            "exit",
            "q",
        }:

            print("Exiting SiagaChat...")
            break

        if not url:

            continue

        try:

            result = predict_url(
                url
            )

            print_result(
                result
            )

        except Exception as exc:

            print()
            print(
                f"Error: {exc}"
            )
            print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()