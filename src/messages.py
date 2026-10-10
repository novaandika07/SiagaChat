"""
SiagaChat reason messages (English + Indonesian).

A "reason" is a small language-neutral item:

    {"code": "suspicious_tld", "params": {"tld": "xyz"}}

The model and the rules only create items. This file turns an item into text
in the chosen language, so the UI can switch language without running the
model again. Scores and labels never depend on this file.

Add a new reason in 3 steps:
  1. add the same code to BOTH "en" and "id" below,
  2. keep the same {placeholders} in both languages,
  3. create it with make("code", placeholder=value).

Check everything with:  python -m src.messages
"""

from __future__ import annotations

import re

MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        # --- rule layer (predict.py) ---
        "official_domain": "{domain} is listed as an official domain.",
        "trusted_suffix": (
            "The domain ending (.go.id / .mil.id) can only be held by "
            "government or military bodies."
        ),
        "popular_domain": (
            "{domain} is a very popular site. Only the domain name is "
            "checked, not the page content."
        ),
        # --- brand rules ---
        "brand_impersonation": (
            "The name '{name}' is used on the domain '{domain}', which is "
            "not an official domain. This is a typical sign of an "
            "impersonation scam."
        ),
        "brand_impersonation_official": (
            "The name '{name}' is used on the domain '{domain}', which is "
            "not an official domain (official site: {official}). This is a "
            "typical sign of an impersonation scam."
        ),
        "brand_in_path": (
            "The name '{brand}' appears in the page address, but the domain "
            "'{domain}' does not belong to them."
        ),
        # --- link shape ---
        "ip_host": (
            "The link uses an IP address instead of a site name. Official "
            "sites almost never do this."
        ),
        "shortener": "The link is shortened, so its real destination is hidden.",
        "suspicious_tld": (
            "The domain ending .{tld} is often used for cheap scam sites."
        ),
        "has_at": (
            "There is an '@' symbol in the address, which can disguise the "
            "real destination."
        ),
        "punycode": (
            "The domain uses special characters (punycode) that can look "
            "like normal letters."
        ),
        "suspicious_words": "Contains words often used by scammers: {words}.",
        "many_subdomains": "The address has too many subdomain layers.",
        "many_hyphens": "The domain name contains many hyphens.",
        "random_name": (
            "The domain name looks random, not like a brand name or a real "
            "word."
        ),
        "many_digits": "The domain name contains many digits.",
        "long_domain": "The domain name is very long.",
        "unparsable_url": (
            "The link format is unusual, so it is hard to analyze."
        ),
        # --- fallbacks (predict.py) ---
        "model_only": (
            "The domain name pattern resembles the scam data the model "
            "learned from, although no single feature stands out."
        ),
        "no_signs": "No suspicious signs were found in the domain name.",
    },
    "id": {
        "official_domain": "Domain {domain} terdaftar sebagai domain resmi.",
        "trusted_suffix": (
            "Akhiran domain (.go.id / .mil.id) hanya bisa dimiliki instansi "
            "pemerintah atau militer."
        ),
        "popular_domain": (
            "{domain} termasuk situs yang sangat populer. Yang diperiksa "
            "hanya nama domain, bukan isi halamannya."
        ),
        "brand_impersonation": (
            "Nama '{name}' dipakai pada domain '{domain}' yang bukan domain "
            "resmi. Ini ciri khas penipuan yang menyamar."
        ),
        "brand_impersonation_official": (
            "Nama '{name}' dipakai pada domain '{domain}' yang bukan domain "
            "resmi (alamat resmi: {official}). Ini ciri khas penipuan yang "
            "menyamar."
        ),
        "brand_in_path": (
            "Nama '{brand}' muncul di alamat halaman, tetapi domain "
            "'{domain}' bukan milik mereka."
        ),
        "ip_host": (
            "Link memakai alamat IP, bukan nama situs. Situs resmi hampir "
            "tidak pernah begini."
        ),
        "shortener": "Link dipendekkan sehingga tujuan aslinya tersembunyi.",
        "suspicious_tld": (
            "Akhiran domain .{tld} sering dipakai untuk situs penipuan murah."
        ),
        "has_at": (
            "Ada simbol '@' di alamat, yang bisa menyamarkan tujuan "
            "sebenarnya."
        ),
        "punycode": (
            "Domain memakai karakter khusus (punycode) yang bisa menyerupai "
            "huruf biasa."
        ),
        "suspicious_words": "Mengandung kata yang sering dipakai penipu: {words}.",
        "many_subdomains": "Alamat memiliki terlalu banyak lapisan subdomain.",
        "many_hyphens": "Nama domain memuat banyak tanda hubung.",
        "random_name": "Nama domain tampak acak, bukan nama merek atau kata.",
        "many_digits": "Nama domain banyak berisi angka.",
        "long_domain": "Nama domain sangat panjang.",
        "unparsable_url": (
            "Format link tidak standar sehingga sulit dianalisis."
        ),
        "model_only": (
            "Pola nama domain mirip data penipuan yang dipelajari model, "
            "meski tidak ada ciri tunggal yang menonjol."
        ),
        "no_signs": "Tidak ditemukan ciri mencurigakan pada nama domain.",
    },
}

DEFAULT_LANG = "en"

# Reasons that only describe a trusted domain. They never explain why a link
# looks dangerous (predict.py uses this to decide when to add 'model_only').
NEUTRAL_CODES = frozenset({"official_domain", "trusted_suffix"})


def make(code: str, **params) -> dict:
    """Create a language-neutral reason item."""
    return {"code": code, "params": params}


def render(item: dict, lang: str = DEFAULT_LANG) -> str:
    """Turn one reason item into text. Never raises."""
    table = MESSAGES.get(lang, MESSAGES[DEFAULT_LANG])
    code = item.get("code", "")
    template = table.get(code) or MESSAGES[DEFAULT_LANG].get(code) or code
    try:
        return template.format(**item.get("params", {}))
    except (KeyError, IndexError):
        return template


def render_all(items: list[dict], lang: str = DEFAULT_LANG) -> list[str]:
    return [render(item, lang) for item in items]


# ============================================================
# SELF TEST:  python -m src.messages
# ============================================================

def _placeholders(text: str) -> set[str]:
    return set(re.findall(r"{(\w+)}", text))


def _self_test() -> None:
    en, idn = MESSAGES["en"], MESSAGES["id"]

    assert en.keys() == idn.keys(), (
        f"codes differ: {sorted(set(en) ^ set(idn))}"
    )
    for code in en:
        assert _placeholders(en[code]) == _placeholders(idn[code]), (
            f"placeholders differ for '{code}'"
        )

    text = render(make("suspicious_tld", tld="xyz"), "en")
    assert text == "The domain ending .xyz is often used for cheap scam sites."
    assert render(make("suspicious_tld", tld="xyz"), "id").startswith("Akhiran")
    assert render(make("no_such_code"), "en") == "no_such_code"
    assert render(make("suspicious_tld"), "en")  # missing param: no crash

    print(f"messages.py: all self tests passed ({len(en)} reasons x 2 languages)")


if __name__ == "__main__":
    _self_test()