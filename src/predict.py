"""
SiagaChat predictor.

    python -m src.predict

Final score = ML model score, adjusted by transparent rules:

    official Indonesian domain (config)      -> capped low
    popular domain (Tranco top 10k) + its own subdomains -> capped low
    .go.id / .mil.id                         -> capped low
    imitates an Indonesian brand             -> raised to Bahaya
    IP address / URL shortener               -> raised to at least Waspada

The URL is never opened or downloaded. Only its text is analysed.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
import tldextract

from src.features import (
    FEATURE_NAMES,
    analyze_brand,
    explain_features_items,
    extract_features,
    has_trusted_suffix,
    parse_url,
)
from src.messages import NEUTRAL_CODES, make, render_all

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "siagachat_url_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "siagachat_url_model_metadata.json"
TRANCO_PATH = BASE_DIR / "data" / "tranco.csv"

SAFE_BELOW = 30
DANGER_FROM = 70

TRUSTED_TOP_N = 10_000  # how many Tranco domains count as "popular"

# Second extractor that also knows shared-hosting suffixes such as
# blogspot.com or github.io. There every customer gets a subdomain, so a
# subdomain is NOT a sign of trust. Used only for the trust rule below;
# features.py and the trained model are untouched.
_HOSTING_EXTRACTOR = tldextract.TLDExtract(
    suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True
)

CAP_OFFICIAL = 10
CAP_TRUSTED = 20
FLOOR_TYPOSQUAT = 85
FLOOR_PATH_BRAND = 45
FLOOR_IP = 55
FLOOR_SHORTENER = 40

ADVICE = {
    "Aman": (
        "Tidak ada ciri penipuan yang terdeteksi pada nama domain. Tetap "
        "berhati-hati: hasil ini bukan jaminan, dan isi halaman tidak diperiksa."
    ),
    "Waspada": (
        "Ada beberapa tanda yang perlu dicurigai. Jangan isi data pribadi, OTP, "
        "atau PIN. Buka situs resminya langsung lewat aplikasi atau pencarian."
    ),
    "Bahaya": (
        "Jangan klik, jangan isi data apa pun, dan jangan install file dari "
        "link ini. Jika sudah terlanjur, segera hubungi bank atau layanan "
        "resminya lewat kanal resmi dan ganti password."
    ),
}


# ============================================================
# LOADING (cached: loaded once, not on every prediction)
# ============================================================

@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run: python -m src.prepare_data  then  "
            "python -m src.train"
        )

    package = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    if metadata.get("feature_names") != FEATURE_NAMES:
        raise ValueError(
            "The saved model was trained with different features than "
            "features.py. Run prepare_data and train again."
        )
    return package["model"], package.get("scaler"), metadata


@lru_cache(maxsize=1)
def _trusted_domains() -> frozenset:
    """Top Tranco domains, if data/tranco.csv exists."""
    if not TRANCO_PATH.exists():
        return frozenset()
    raw = pd.read_csv(TRANCO_PATH, header=None, dtype=str).dropna()
    domains = raw.iloc[:, -1].astype(str).str.strip().str.lower()
    return frozenset(domains[domains != "domain"].head(TRUSTED_TOP_N))


# ============================================================
# MODEL SCORE
# ============================================================

def model_score(url: str) -> float:
    """Malicious probability from the ML model, 0-100."""
    model, scaler, _ = _load_model()
    features = extract_features(url)
    X = pd.DataFrame([features], columns=FEATURE_NAMES)
    X_in = scaler.transform(X) if scaler is not None else X
    proba = model.predict_proba(X_in)[0][list(model.classes_).index(1)]
    return float(proba) * 100.0


# ============================================================
# RULE LAYER
# ============================================================

def _label(score: float) -> str:
    if score < SAFE_BELOW:
        return "Aman"
    if score < DANGER_FROM:
        return "Waspada"
    return "Bahaya"


def _is_trusted_popular(parsed: dict) -> bool:
    """
    True for a popular domain (Tranco top N) and its own subdomains, e.g.
    docs.python.org. Subdomains on shared-hosting platforms are NOT trusted:
    'evil.blogspot.com' is not safe just because 'blogspot.com' is popular.
    """
    registered = parsed["registered"]
    if registered not in _trusted_domains():
        return False

    host = parsed["host_clean"]
    if host == registered:
        return True

    # On a shared-hosting suffix (blogspot.com, github.io, ...) this extractor
    # treats the platform as the suffix, so the customer name becomes the
    # registered domain and no longer equals 'registered'.
    ext = _HOSTING_EXTRACTOR(host)
    hosting_registered = (
        f"{ext.domain}.{ext.suffix}" if ext.domain and ext.suffix else host
    )
    return hosting_registered == registered


# ============================================================
# RULE LAYER (continued)
# ============================================================

def apply_rules(score: float, url: str) -> tuple[float, list[dict], list[str]]:
    """Return (adjusted score, rule reason items, rules fired)."""
    parsed = parse_url(url)
    brand = analyze_brand(url)
    features = extract_features(url)

    reasons: list[dict] = []
    fired: list[str] = []

    if brand["is_official"]:
        fired.append("official_domain")
        reasons.append(make("official_domain", domain=parsed["registered"]))
        return min(score, CAP_OFFICIAL), reasons, fired

    if has_trusted_suffix(parsed["host_clean"]):
        fired.append("trusted_suffix")
        reasons.append(make("trusted_suffix"))
        return min(score, CAP_TRUSTED), reasons, fired

    if brand["is_typosquatting"]:
        fired.append("brand_impersonation")
        return max(score, FLOOR_TYPOSQUAT), reasons, fired

    if _is_trusted_popular(parsed):
        fired.append("popular_domain")
        reasons.append(make("popular_domain", domain=parsed["registered"]))
        return min(score, CAP_TRUSTED), reasons, fired

    if brand["brand_in_path"] and features["suspicious_word_count"] > 0:
        fired.append("brand_in_path")
        score = max(score, FLOOR_PATH_BRAND)
    if features["is_ip"]:
        fired.append("ip_host")
        score = max(score, FLOOR_IP)
    if features["is_shortener"]:
        fired.append("shortener")
        score = max(score, FLOOR_SHORTENER)

    return score, reasons, fired


# ============================================================
# PUBLIC API
# ============================================================

def predict_url(url: str) -> dict:
    """Analyse one URL. Raises ValueError for unusable input."""
    url = str(url).strip()
    if not url:
        raise ValueError("URL tidak boleh kosong.")

    base = model_score(url)
    final, rule_reasons, fired = apply_rules(base, url)
    final = round(max(0.0, min(100.0, final)), 2)
    label = _label(final)

    items = rule_reasons + explain_features_items(url)

    # Keep reasons consistent with the verdict.
    if label == "Aman" and not items:
        items.append(make("no_signs"))
    if label != "Aman" and not any(
        item["code"] not in NEUTRAL_CODES for item in items
    ):
        items.append(make("model_only"))

    return {
        "url": url,
        "risk_score": final,
        "model_score": round(base, 2),
        "label": label,
        # Indonesian text, kept for the command-line checker.
        "reasons": render_all(items, "id"),
        # Language-neutral items: the UI renders them in the chosen language
        # with src.messages.render().
        "reason_items": items,
        "rules_fired": fired,
        "advice": ADVICE[label],
    }


def _print(result: dict) -> None:
    print()
    print("=" * 62)
    print(f"URL     : {result['url']}")
    print(f"Status  : {result['label']}   (skor {result['risk_score']}/100)")
    print(f"Model   : {result['model_score']}   Aturan: {result['rules_fired'] or '-'}")
    print("Alasan:")
    for reason in result["reasons"]:
        print(f"  - {reason}")
    print(f"Saran   : {result['advice']}")
    print("=" * 62)


def main() -> None:
    print("SiagaChat URL checker. Ketik 'q' untuk keluar.")
    while True:
        try:
            url = input("\nURL: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if url.lower() in {"q", "quit", "exit"}:
            break
        if not url:
            continue
        try:
            _print(predict_url(url))
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()