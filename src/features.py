"""
Feature extraction module for SiagaChat.

Important design:
- HTTP/HTTPS scheme is NOT used as an ML feature.
- All ML features are extracted from the URL without http:// or https://.
- HTTPS/HTTP may still be reported by explain_features().
- Feature names and order are defined in FEATURE_NAMES.
"""

from __future__ import annotations

import ipaddress
import math
from collections import Counter
from urllib.parse import urlsplit

import tldextract
from rapidfuzz import fuzz

from src import config


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
# FAILURE STATISTICS
# ============================================================

FEATURE_EXTRACTION_FAILURES = 0
FEATURE_EXTRACTION_ERROR_TYPES = Counter()


def get_feature_extraction_failures() -> int:
    """Return total number of failed feature extractions."""
    return FEATURE_EXTRACTION_FAILURES


def get_feature_extraction_error_types() -> dict:
    """Return extraction failure counts grouped by exception type."""
    return dict(FEATURE_EXTRACTION_ERROR_TYPES)


def reset_feature_extraction_failures() -> None:
    """Reset feature extraction failure statistics."""
    global FEATURE_EXTRACTION_FAILURES

    FEATURE_EXTRACTION_FAILURES = 0
    FEATURE_EXTRACTION_ERROR_TYPES.clear()


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_url(url: str) -> str:
    """
    Normalize URL for ML feature extraction.

    The scheme is deliberately removed.

    Examples:
        https://example.com/login
        -> example.com/login

        http://example.com/login
        -> example.com/login
    """

    if url is None:
        return ""

    url = str(url).strip().lower()

    if url.startswith("https://"):
        return url[8:]

    if url.startswith("http://"):
        return url[7:]

    return url


def _parse_scheme(url: str) -> str:
    """Return the original URL scheme for explanations only."""

    value = str(url).strip().lower()

    if value.startswith("https://"):
        return "https"

    if value.startswith("http://"):
        return "http"

    return ""


def _parse_url(normalized_url: str):
    """
    Parse a scheme-less URL safely.

    // is used only internally by urlsplit().
    It is NOT included in feature values.
    """

    return urlsplit("//" + normalized_url)


# ============================================================
# ENTROPY
# ============================================================

def calculate_entropy(text: str) -> float:
    """Calculate Shannon entropy."""

    if not text:
        return 0.0

    counts = Counter(text)
    length = len(text)

    entropy = 0.0

    for count in counts.values():
        probability = count / length
        entropy -= probability * math.log2(probability)

    return entropy


# Backward-compatible private name.
_calculate_entropy = calculate_entropy


# ============================================================
# HOSTNAME / DOMAIN HELPERS
# ============================================================

def _hostname_tokens(hostname: str) -> list[str]:
    """
    Split hostname into whole words using dots and hyphens.

    Example:
        bca.login-aman.xyz

    becomes:

        ["bca", "login", "aman", "xyz"]
    """

    if not hostname:
        return []

    tokens = []

    for label in hostname.lower().split("."):
        for token in label.split("-"):
            token = token.strip()

            if token:
                tokens.append(token)

    return tokens


def _is_ip_address(hostname: str) -> bool:
    """Return True when hostname is a valid IPv4 or IPv6 address."""

    if not hostname:
        return False

    try:
        ipaddress.ip_address(hostname.strip("[]"))
        return True
    except ValueError:
        return False


def _registered_domain(extracted) -> str:
    """
    Return registrable domain using current tldextract API.

    Example:
        login.bca.co.id
        -> bca.co.id
    """

    value = extracted.top_domain_under_public_suffix

    if value:
        return value.lower()

    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}".lower()

    return extracted.domain.lower()


# ============================================================
# OFFICIAL DOMAIN CACHE
# ============================================================

def _build_official_domain_cache() -> dict:
    """
    Parse official domains once at module load.

    This avoids repeatedly calling tldextract for official domains
    inside every URL prediction.
    """

    cache = {}

    for brand, domains in config.OFFICIAL_DOMAINS.items():

        brand_key = str(brand).strip().lower()

        if not brand_key:
            continue

        cache[brand_key] = []

        for official_domain in domains:

            official_domain = (
                str(official_domain)
                .strip()
                .lower()
                .rstrip(".")
            )

            if not official_domain:
                continue

            extracted = tldextract.extract(official_domain)

            cache[brand_key].append({
                "domain": official_domain,
                "registered_domain": _registered_domain(extracted),
                "hostname_tokens": frozenset(
                    _hostname_tokens(official_domain)
                ),
            })

    return cache


# IMPORTANT:
# _hostname_tokens and _registered_domain are defined BEFORE
# this function is called.
OFFICIAL_DOMAIN_CACHE = _build_official_domain_cache()


# ============================================================
# BRAND ANALYSIS
# ============================================================

def _analyze_brands(hostname: str, registered_domain: str):
    """
    Analyze brand mentions in hostname.

    Returns:
        is_typosquatting, max_brand_similarity
    """

    hostname_tokens = set(_hostname_tokens(hostname))

    if not hostname_tokens:
        return 0, 0.0

    is_typosquatting = 0
    max_similarity = 0.0

    # Compare the registrable domain against each configured brand.
    domain_name = registered_domain.split(".")[0]

    for brand, official_domains in OFFICIAL_DOMAIN_CACHE.items():

        # ----------------------------------------------------
        # Exact whole-word brand detection
        # ----------------------------------------------------

        brand_present = brand in hostname_tokens

        # ----------------------------------------------------
        # Similarity
        #
        # Compare the actual domain name against the brand.
        # This catches things like:
        #   bcca
        #   bca-login
        #   bankbca
        # ----------------------------------------------------

        similarity = fuzz.ratio(domain_name, brand)

        if similarity > max_similarity:
            max_similarity = float(similarity)

        # ----------------------------------------------------
        # Official domain matching
        # ----------------------------------------------------

        if brand_present:

            official_registered_domains = {
                item["registered_domain"]
                for item in official_domains
            }

            if registered_domain not in official_registered_domains:
                is_typosquatting = 1

        # ----------------------------------------------------
        # Brand in subdomain with unrelated main domain
        #
        # Example:
        #   bca.login-aman.xyz
        #
        # registered_domain:
        #   login-aman.xyz
        #
        # => suspicious
        # ----------------------------------------------------

        if brand_present:

            hostname_is_official = False

            for official in official_domains:

                official_domain = official["domain"]

                if hostname == official_domain:
                    hostname_is_official = True
                    break

                if hostname.endswith("." + official_domain):
                    hostname_is_official = True
                    break

            if not hostname_is_official:
                is_typosquatting = 1

    return is_typosquatting, max_similarity


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def _empty_features() -> dict:
    """Return a complete feature dictionary with safe defaults."""

    return {
        "url_length": 0,
        "domain_length": 0,
        "path_length": 0,
        "num_dots": 0,
        "num_hyphens": 0,
        "num_at": 0,
        "num_question": 0,
        "num_equals": 0,
        "num_ampersand": 0,
        "num_percent": 0,
        "num_digits": 0,
        "num_non_alphanum": 0,
        "num_subdomains": 0,
        "is_ip": 0,
        "suspicious_word_count": 0,
        "is_shortener": 0,
        "is_suspicious_tld": 0,
        "domain_entropy": 0.0,
        "digit_ratio": 0.0,
        "is_typosquatting": 0,
        "max_brand_similarity": 0.0,
    }


def extract_features(url: str) -> dict:
    """
    Extract all ML features from a URL.

    HTTP/HTTPS is deliberately excluded from the feature vector.
    """

    global FEATURE_EXTRACTION_FAILURES

    features = _empty_features()

    try:

        normalized_url = normalize_url(url)

        if not normalized_url:
            return features

        parsed = _parse_url(normalized_url)

        hostname = (parsed.hostname or "").lower()

        # ----------------------------------------------------
        # Basic lengths
        # ----------------------------------------------------

        features["url_length"] = len(normalized_url)
        features["path_length"] = len(parsed.path)

        # ----------------------------------------------------
        # Character counts
        # ----------------------------------------------------

        features["num_dots"] = normalized_url.count(".")
        features["num_hyphens"] = normalized_url.count("-")
        features["num_at"] = normalized_url.count("@")
        features["num_question"] = normalized_url.count("?")
        features["num_equals"] = normalized_url.count("=")
        features["num_ampersand"] = normalized_url.count("&")
        features["num_percent"] = normalized_url.count("%")
        features["num_digits"] = sum(
            char.isdigit()
            for char in normalized_url
        )
        features["num_non_alphanum"] = sum(
            not char.isalnum()
            for char in normalized_url
        )

        # ----------------------------------------------------
        # Domain parsing
        # ----------------------------------------------------

        extracted = tldextract.extract(hostname)

        registered_domain = _registered_domain(extracted)

        features["domain_length"] = len(registered_domain)

        features["num_subdomains"] = (
            len(extracted.subdomain.split("."))
            if extracted.subdomain
            else 0
        )

        # ----------------------------------------------------
        # IP address
        # ----------------------------------------------------

        features["is_ip"] = int(
            _is_ip_address(hostname)
        )

        # ----------------------------------------------------
        # Suspicious words
        # ----------------------------------------------------

        features["suspicious_word_count"] = sum(
            1
            for word in config.SUSPICIOUS_WORDS
            if str(word).lower() in normalized_url
        )

        # ----------------------------------------------------
        # Shortener
        # ----------------------------------------------------

        hostname_without_www = hostname

        if hostname_without_www.startswith("www."):
            hostname_without_www = hostname_without_www[4:]

        if (
            hostname_without_www in config.SHORTENERS
            or registered_domain in config.SHORTENERS
        ):
            features["is_shortener"] = 1

        # ----------------------------------------------------
        # Suspicious TLD
        # ----------------------------------------------------

        suffix = extracted.suffix.lower()

        if suffix in {
            str(tld).lower().lstrip(".")
            for tld in config.SUSPICIOUS_TLDS
        }:
            features["is_suspicious_tld"] = 1

        # ----------------------------------------------------
        # Entropy
        # ----------------------------------------------------

        domain_name = extracted.domain.lower()

        features["domain_entropy"] = calculate_entropy(
            domain_name
        )

        # ----------------------------------------------------
        # Digit ratio
        # ----------------------------------------------------

        if domain_name:
            features["digit_ratio"] = (
                sum(char.isdigit() for char in domain_name)
                / len(domain_name)
            )

        # ----------------------------------------------------
        # Brand / typosquatting
        # ----------------------------------------------------

        (
            features["is_typosquatting"],
            features["max_brand_similarity"],
        ) = _analyze_brands(
            hostname,
            registered_domain,
        )

    except Exception as exc:

        FEATURE_EXTRACTION_FAILURES += 1
 
        FEATURE_EXTRACTION_ERROR_TYPES[
            type(exc).__name__
        ] += 1

    return features


# ============================================================
# EXPLANATION
# ============================================================

def explain_features(url: str) -> list[str]:
    """
    Generate human-readable reasons.

    HTTPS/HTTP is allowed here because this function is NOT used
    as an ML feature extractor.
    """

    reasons = []

    try:

        original_url = str(url).strip()

        if not original_url:
            return ["URL kosong"]

        scheme = _parse_scheme(original_url)

        normalized_url = normalize_url(original_url)

        parsed = _parse_url(normalized_url)

        hostname = (parsed.hostname or "").lower()

        extracted = tldextract.extract(hostname)

        registered_domain = _registered_domain(extracted)

        # ----------------------------------------------------
        # HTTP explanation only
        # ----------------------------------------------------

        if scheme == "http":
            reasons.append(
                "Uses insecure HTTP protocol (not HTTPS)"
            )

        # ----------------------------------------------------
        # @ symbol
        # ----------------------------------------------------

        if "@" in normalized_url:
            reasons.append(
                "Contains '@' symbol (often used to hide the real domain)"
            )

        # ----------------------------------------------------
        # Suspicious words
        # ----------------------------------------------------

        found_words = [
            str(word)
            for word in config.SUSPICIOUS_WORDS
            if str(word).lower() in normalized_url
        ]

        if found_words:
            reasons.append(
                "Contains suspicious words: "
                + ", ".join(found_words[:3])
            )

        # ----------------------------------------------------
        # Suspicious TLD
        # ----------------------------------------------------

        suffix = extracted.suffix.lower()

        if suffix in {
            str(tld).lower().lstrip(".")
            for tld in config.SUSPICIOUS_TLDS
        }:
            reasons.append(
                f"Uses suspicious top-level domain (.{suffix})"
            )

        # ----------------------------------------------------
        # IP
        # ----------------------------------------------------

        if _is_ip_address(hostname):
            reasons.append(
                "Uses a raw IP address instead of a domain name"
            )

        # ----------------------------------------------------
        # Shortener
        # ----------------------------------------------------

        hostname_without_www = hostname

        if hostname_without_www.startswith("www."):
            hostname_without_www = hostname_without_www[4:]

        if (
            hostname_without_www in config.SHORTENERS
            or registered_domain in config.SHORTENERS
        ):
            reasons.append(
                "Uses a URL shortener to hide the destination"
            )

        # ----------------------------------------------------
        # Brand impersonation
        # ----------------------------------------------------

        hostname_tokens = set(
            _hostname_tokens(hostname)
        )

        for brand, official_domains in OFFICIAL_DOMAIN_CACHE.items():

            if brand not in hostname_tokens:
                continue

            official_registered_domains = {
                item["registered_domain"]
                for item in official_domains
            }

            if registered_domain not in official_registered_domains:

                reasons.append(
                    f"Mimics brand '{brand}' but uses "
                    f"an unofficial domain ({registered_domain})"
                )

                break

        # ----------------------------------------------------
        # No suspicious explanation
        # ----------------------------------------------------

        return reasons

    except Exception:
        return [
            "URL format is invalid or could not be analyzed"
        ]


# ============================================================
# UNIT TESTS
# ============================================================

def _run_tests():

    print("Running unit tests for features.py...")

    # --------------------------------------------------------
    # Test 1: HTTP and HTTPS must have identical ML features
    # --------------------------------------------------------

    http_features = extract_features(
        "http://example.com/login"
    )

    https_features = extract_features(
        "https://example.com/login"
    )

    assert http_features == https_features

    # --------------------------------------------------------
    # Test 2: Official BCA
    # --------------------------------------------------------

    bca = extract_features(
        "https://bca.co.id/login"
    )

    assert bca["is_typosquatting"] == 0

    # --------------------------------------------------------
    # Test 3: Brand impersonation
    # --------------------------------------------------------

    phishing = extract_features(
        "http://bca-login-secure.xyz/verify"
    )

    assert phishing["is_typosquatting"] == 1
    assert phishing["is_suspicious_tld"] == 1

    # --------------------------------------------------------
    # Test 4: Brand in unrelated subdomain
    # --------------------------------------------------------

    subdomain_attack = extract_features(
        "https://bca.login-aman.xyz"
    )

    assert subdomain_attack["is_typosquatting"] == 1

    # --------------------------------------------------------
    # Test 5: IP
    # --------------------------------------------------------

    ip_features = extract_features(
        "http://192.168.1.1/admin"
    )

    assert ip_features["is_ip"] == 1

    # --------------------------------------------------------
    # Test 6: Shortener
    # --------------------------------------------------------

    shortener = extract_features(
        "https://bit.ly/klaim-hadiah"
    )

    assert shortener["is_shortener"] == 1

    # --------------------------------------------------------
    # Test 7: No HTTPS feature
    # --------------------------------------------------------

    assert "is_https" not in http_features

    # --------------------------------------------------------
    # Test 8: Exact feature schema
    # --------------------------------------------------------

    assert list(http_features.keys()) == FEATURE_NAMES

    # --------------------------------------------------------
    # Test 9: Explanation still mentions HTTP
    # --------------------------------------------------------

    reasons = explain_features(
        "http://example.com/login"
    )

    assert any(
        "HTTP" in reason
        for reason in reasons
    )

    print("All feature tests passed successfully! 🎉")


if __name__ == "__main__":
    _run_tests()