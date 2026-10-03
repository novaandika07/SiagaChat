"""
SiagaChat feature extraction.

Design (important for the README):

1. ML features describe the HOST (domain name) of a URL, plus a few
   path-independent signals. Path, query and scheme are deliberately NOT used
   as ML features, because in the training dataset they are shortcuts:
     - only ~8% of benign URLs have http(s):// vs 40% of malicious ones
     - URLs without a path are 98.8% malicious (benign ones nearly always
       have a path)
   A model trained on those would call tokopedia.com "dangerous".

2. Indonesian brand knowledge (typosquatting, official domains) is a separate
   rule layer (analyze_brand). The public dataset has almost no Indonesian
   brands, so a model cannot learn them; explicit rules can.

No function here opens or downloads a URL. Everything is computed from text.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlsplit

import tldextract
from rapidfuzz import fuzz, process

from src import config

# Offline extractor: uses the public-suffix snapshot bundled with tldextract,
# so no network access (and no random failures) during bulk processing.
_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)


# ============================================================
# FEATURE SCHEMA (single source of truth)
# ============================================================

FEATURE_NAMES = [
    "host_length",
    "domain_name_length",
    "longest_label_length",
    "subdomain_count",
    "num_dots_host",
    "num_hyphens_host",
    "num_digits_host",
    "digit_ratio_host",
    "domain_entropy",
    "max_consonant_run",
    "is_ip",
    "is_shortener",
    "is_suspicious_tld",
    "suspicious_word_count",
    "has_at",
    "has_punycode",
    "has_port",
]


# ============================================================
# PRECOMPUTED LOOKUPS
# ============================================================

_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.\-]*://")
_TOKEN_RE = re.compile(r"[^a-z0-9]+")
_HOST_TOKEN_RE = re.compile(r"[.\-_]+")
_CONSONANT_RUN_RE = re.compile(r"[bcdfghjklmnpqrstvwxyz]+")

_SHORTENERS = frozenset(s.lower() for s in config.SHORTENERS)
_SUSPICIOUS_TLDS = frozenset(t.lower() for t in config.SUSPICIOUS_TLDS)
_SHORT_WORDS = frozenset(
    w.lower() for w in config.SUSPICIOUS_WORDS if len(w) < 6
)
_LONG_WORDS = tuple(w.lower() for w in config.SUSPICIOUS_WORDS if len(w) >= 6)
_BRANDS = tuple(b.lower() for b in config.BRAND_KEYWORDS)
_BRANDS_FUZZY = tuple(b for b in _BRANDS if len(b) >= 5)
_OFFICIAL = tuple(config.ALL_OFFICIAL_DOMAINS)
_TRUSTED_SUFFIXES = tuple(config.TRUSTED_SUFFIXES)


# ============================================================
# PARSING
# ============================================================

def parse_url(url: str) -> dict:
    """
    Split a URL into the parts we need. Raises ValueError for unusable input.

    The scheme and a leading 'www.' are removed so that 'tokopedia.com',
    'http://tokopedia.com' and 'https://www.tokopedia.com' look identical.
    """
    raw = str(url).strip().lower()
    if not raw:
        raise ValueError("empty url")

    no_scheme = _SCHEME_RE.sub("", raw)
    parts = urlsplit("//" + no_scheme)  # may raise ValueError

    host = (parts.hostname or "").strip(".")
    if not host:
        raise ValueError("no host")

    try:
        has_port = parts.port is not None
    except ValueError:
        has_port = False

    host_clean = host[4:] if host.startswith("www.") else host

    try:
        ipaddress.ip_address(host_clean.strip("[]"))
        is_ip = True
    except ValueError:
        is_ip = False

    ext = _EXTRACTOR(host_clean)
    # Built by hand: 'registered_domain' is deprecated in newer tldextract.
    registered = (
        f"{ext.domain}.{ext.suffix}" if ext.domain and ext.suffix else host_clean
    )

    authority = no_scheme.split("/", 1)[0]
    path = parts.path + (("?" + parts.query) if parts.query else "")

    return {
        "raw": raw,
        "no_scheme": no_scheme,
        "authority": authority,
        "host": host,
        "host_clean": host_clean,
        "ext": ext,
        "registered": registered,
        "is_ip": is_ip,
        "has_port": has_port,
        "path": path,
    }


def get_registered_domain(url: str) -> str:
    """Group key used to split train/test without sharing a domain."""
    return parse_url(url)["registered"]


def calculate_entropy(text: str) -> float:
    """Shannon entropy: higher means more random-looking."""
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def count_suspicious_words(no_scheme_url: str) -> int:
    """
    Count scam-style words. Short words must be a whole token (so 'pin' does
    not match 'pinterest'); long words may appear as a substring.
    """
    tokens = set(_TOKEN_RE.split(no_scheme_url))
    count = sum(1 for w in _SHORT_WORDS if w in tokens)
    count += sum(
        1 for w in _LONG_WORDS if w in tokens or w in no_scheme_url
    )
    return count


def is_official_host(host: str) -> bool:
    """True if host is (a subdomain of) a configured official domain."""
    host = host.lower().strip(".")
    return any(host == d or host.endswith("." + d) for d in _OFFICIAL)


def has_trusted_suffix(host: str) -> bool:
    """True for suffixes that are restricted (e.g. go.id government sites)."""
    host = host.lower().strip(".")
    return any(host.endswith("." + s) or host == s for s in _TRUSTED_SUFFIXES)


# ============================================================
# ML FEATURES
# ============================================================

def extract_features(url: str) -> dict:
    """
    Return {feature_name: value} for FEATURE_NAMES.

    Raises ValueError if the URL cannot be parsed. We never return silent
    zeros, so bad input cannot sneak into the training data.
    """
    p = parse_url(url)
    ext = p["ext"]
    host = p["host_clean"]
    domain_name = ext.domain or host

    subdomain_count = (
        len([s for s in ext.subdomain.split(".") if s]) if ext.subdomain else 0
    )
    labels = [label for label in host.split(".") if label]
    digits_in_host = sum(ch.isdigit() for ch in host)
    consonant_runs = _CONSONANT_RUN_RE.findall(domain_name)
    tld = ext.suffix.split(".")[-1] if ext.suffix else ""

    return {
        "host_length": len(host),
        "domain_name_length": len(domain_name),
        "longest_label_length": max((len(x) for x in labels), default=0),
        "subdomain_count": subdomain_count,
        "num_dots_host": host.count("."),
        "num_hyphens_host": host.count("-"),
        "num_digits_host": digits_in_host,
        "digit_ratio_host": digits_in_host / len(host) if host else 0.0,
        "domain_entropy": calculate_entropy(domain_name),
        "max_consonant_run": max((len(r) for r in consonant_runs), default=0),
        "is_ip": int(p["is_ip"]),
        "is_shortener": int(
            host in _SHORTENERS or p["registered"] in _SHORTENERS
        ),
        "is_suspicious_tld": int(tld in _SUSPICIOUS_TLDS),
        "suspicious_word_count": count_suspicious_words(p["no_scheme"]),
        "has_at": int("@" in p["authority"]),
        "has_punycode": int("xn--" in host),
        "has_port": int(p["has_port"]),
    }


def extract_features_and_group(url: str) -> tuple[dict, str]:
    """Features plus the registered domain (used for grouped splitting)."""
    p = parse_url(url)
    return extract_features(url), p["registered"]


# ============================================================
# BRAND / TYPOSQUATTING RULES (not part of the ML model)
# ============================================================

def analyze_brand(url: str) -> dict:
    """
    Check whether a URL imitates a known Indonesian brand.

    Returns:
        is_official       host belongs to a configured official domain
        is_typosquatting  looks like a brand but is NOT official
        brand             brand it imitates (or None)
        match_type        'exact_token', 'contains', 'lookalike' or None
        similarity        0-100 (only for 'lookalike')
        brand_in_path     brand name appears in the path of a non-official host
    """
    p = parse_url(url)
    host = p["host_clean"]
    domain_name = p["ext"].domain or host

    result = {
        "is_official": is_official_host(host),
        "is_typosquatting": False,
        "brand": None,
        "match_type": None,
        "similarity": 0.0,
        "brand_in_path": None,
        "registered_domain": p["registered"],
    }

    if result["is_official"] or p["is_ip"]:
        return result

    host_tokens = {t for t in _HOST_TOKEN_RE.split(host) if t}

    # 1) brand appears as a whole word in the host (bca-secure-login.xyz,
    #    bca.login-aman.xyz)
    for brand in _BRANDS:
        if brand in host_tokens:
            result.update(
                is_typosquatting=True, brand=brand, match_type="exact_token"
            )
            return result

    # 2) long brand glued to other letters (klik-tokopediaaman.com)
    for brand in _BRANDS_FUZZY:
        if brand in domain_name:
            result.update(
                is_typosquatting=True, brand=brand, match_type="contains"
            )
            return result

    # 3) lookalike spelling of a word in the domain (tokopedla, shopeee).
    #    Only brands with 5+ letters and a strict cutoff: short real words
    #    (and names like "bilibili" vs "blibli") would otherwise match.
    for token in sorted(host_tokens):
        if len(token) < 5:
            continue
        best = process.extractOne(
            token, _BRANDS_FUZZY, scorer=fuzz.ratio, score_cutoff=88
        )
        if best is not None:
            result.update(
                is_typosquatting=True,
                brand=best[0],
                match_type="lookalike",
                similarity=float(best[1]),
            )
            return result

    # 4) brand mentioned in the path of an unrelated domain (weak signal)
    path_tokens = {t for t in _TOKEN_RE.split(p["path"]) if t}
    for brand in _BRANDS:
        if brand in path_tokens:
            result["brand_in_path"] = brand
            break

    return result


# ============================================================
# HUMAN-READABLE EXPLANATION (Indonesian)
# ============================================================

def explain_features(url: str) -> list[str]:
    """List of plain-Indonesian reasons why a URL looks suspicious."""
    reasons: list[str] = []

    try:
        p = parse_url(url)
        feats = extract_features(url)
        brand = analyze_brand(url)
    except ValueError:
        return ["Format link tidak standar sehingga sulit dianalisis."]

    host = p["host_clean"]

    if brand["is_typosquatting"]:
        name = brand["brand"]
        official = config.OFFICIAL_DOMAINS.get(name, [])
        hint = f" (alamat resmi: {official[0]})" if official else ""
        reasons.append(
            f"Nama '{name}' dipakai pada domain '{p['registered']}' yang "
            f"bukan domain resmi{hint}. Ini ciri khas penipuan yang menyamar."
        )
    elif brand["brand_in_path"] and feats["suspicious_word_count"] > 0:
        reasons.append(
            f"Nama '{brand['brand']}' muncul di alamat halaman, tetapi domain "
            f"'{p['registered']}' bukan milik mereka."
        )

    if feats["is_ip"]:
        reasons.append(
            "Link memakai alamat IP, bukan nama situs. Situs resmi hampir "
            "tidak pernah begini."
        )
    if feats["is_shortener"]:
        reasons.append(
            "Link dipendekkan sehingga tujuan aslinya tersembunyi."
        )
    if feats["is_suspicious_tld"]:
        tld = p["ext"].suffix.split(".")[-1]
        reasons.append(
            f"Akhiran domain .{tld} sering dipakai untuk situs penipuan "
            "murah."
        )
    if feats["has_at"]:
        reasons.append(
            "Ada simbol '@' di alamat, yang bisa menyamarkan tujuan sebenarnya."
        )
    if feats["has_punycode"]:
        reasons.append(
            "Domain memakai karakter khusus (punycode) yang bisa menyerupai "
            "huruf biasa."
        )

    if feats["suspicious_word_count"] > 0:
        tokens = set(_TOKEN_RE.split(p["no_scheme"]))
        found = [
            w
            for w in config.SUSPICIOUS_WORDS
            if w.lower() in tokens
            or (len(w) >= 6 and w.lower() in p["no_scheme"])
        ]
        reasons.append(
            "Mengandung kata yang sering dipakai penipu: "
            + ", ".join(found[:4])
            + "."
        )

    if feats["subdomain_count"] >= 3:
        reasons.append("Alamat memiliki terlalu banyak lapisan subdomain.")
    if feats["num_hyphens_host"] >= 3:
        reasons.append("Nama domain memuat banyak tanda hubung.")
    if (
        feats["domain_name_length"] >= 10
        and feats["domain_entropy"] >= 3.4
        and feats["max_consonant_run"] >= 5
    ):
        reasons.append("Nama domain tampak acak, bukan nama merek atau kata.")
    if feats["digit_ratio_host"] >= 0.3 and not p["is_ip"]:
        reasons.append("Nama domain banyak berisi angka.")
    if len(host) >= 40:
        reasons.append("Nama domain sangat panjang.")

    return reasons


# ============================================================
# SELF TEST:  python -m src.features
# ============================================================

def _self_test() -> None:
    f = extract_features("https://www.bca.co.id/login")
    assert f["is_shortener"] == 0 and f["is_ip"] == 0

    assert analyze_brand("https://www.bca.co.id/login")["is_official"]
    assert analyze_brand("m.klikbca.com")["is_official"]

    b = analyze_brand("http://bca-secure-login.xyz")
    assert b["is_typosquatting"] and b["brand"] == "bca"

    b = analyze_brand("bca.login-aman.xyz")  # brand in subdomain
    assert b["is_typosquatting"]

    assert not analyze_brand("brilliant.com")["is_typosquatting"]
    assert not analyze_brand("kompas.com")["is_typosquatting"]
    assert analyze_brand("tokopedla-promo.com")["is_typosquatting"]

    assert extract_features("http://192.168.1.1/admin")["is_ip"] == 1
    assert extract_features("bit.ly/klaim-hadiah")["is_shortener"] == 1
    assert extract_features("https://example.com/bit.ly/login")["is_shortener"] == 0
    assert extract_features("shopee-gratis-hadiah.xyz")["suspicious_word_count"] >= 2
    assert extract_features("pinterest.com")["suspicious_word_count"] == 0

    # scheme and www must not change the features
    a = extract_features("https://www.tokopedia.com")
    c = extract_features("tokopedia.com")
    assert a == c, "scheme/www must not change features"

    assert len(explain_features("http://bca-secure-login.xyz")) >= 2

    try:
        extract_features("")
    except ValueError:
        pass
    else:
        raise AssertionError("empty url must raise ValueError")

    print("features.py: all self tests passed")


if __name__ == "__main__":
    _self_test()