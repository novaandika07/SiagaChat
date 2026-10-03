# src/config.py
"""
Configuration file for SiagaChat URL classifier.
Contains brand keywords, suspicious words, shorteners, and official domains.
"""

# List of Indonesian brands often impersonated in phishing attacks
BRAND_KEYWORDS = [
    "bca", "bri", "mandiri", "bni", "cimb", "btn", "danamon", "permata",
    "shopee", "tokopedia", "lazada", "blibli", "bukalapak", "tiktok",
    "jne", "jnt", "jet", "sicepat", "ninja", "anteraja", "pos",
    "gopay", "ovo", "dana", "linkaja", "kredivo", "akulaku",
    "bpjs", "dukcapil", "djp", "pajak", "kemendikbud", "kemenkeu"
]

# Suspicious words often found in malicious URLs (Indonesian & English mix)
SUSPICIOUS_WORDS = [
    "login", "verify", "secure", "update", "confirm", "reset", "unlock",
    "akun", "blokir", "hadiah", "klaim", "gratis", "menang", "undian",
    "resi", "paket", "undangan", "apk", "download", "instal", "dana",
    "cair", "pencairan", "ktp", "otp", "pin", "password", "sandbox"
]

# Common URL shorteners used to hide the real destination
SHORTENERS = [
    "bit.ly", "tinyurl.com", "s.id", "cutt.ly", "short.link", "t.co",
    "ow.ly", "goo.gl", "rebrand.ly", "linktr.ee"
]

# Top-Level Domains (TLDs) frequently abused by scammers
SUSPICIOUS_TLDS = [
    "xyz", "top", "click", "loan", "work", "tk", "ml", "ga", "cf", "gq",
    "online", "site", "space", "win", "vip", "info"
]

# Mapping of brand keywords to a LIST of their official, legitimate domains (lowercase)
OFFICIAL_DOMAINS = {
    "bca": ["bca.co.id", "halobca.com"],
    "bri": ["bri.co.id", "bukaberbagi.bri.co.id"],
    "mandiri": ["mandiri.co.id", "livinbymandiri.co.id"],
    "bni": ["bni.co.id"],
    "cimb": ["cimbniaga.co.id"],
    "shopee": ["shopee.co.id", "shopee.co.th"], # contoh variasi
    "tokopedia": ["tokopedia.com"],
    "lazada": ["lazada.co.id"],
    "jne": ["jne.co.id"],
    "jnt": ["jet.co.id"],
    "gopay": ["gopay.co.id"],
    "ovo": ["ovo.id"],
    "dana": ["dana.id"],
    "bpjs": ["bpjs-kesehatan.go.id", "bpjsketenagakerjaan.go.id"],
    "pajak": ["pajak.go.id", "djponline.pajak.go.id"],
    "dukcapil": ["dukcapil.kemendagri.go.id"]
}

# PRECOMPUTATION: Flatten all official domains into a single set for O(1) fast lookup
# This prevents calling tldextract or doing slow list searches inside the URL loop.
ALL_OFFICIAL_DOMAINS = set()
for domains_list in OFFICIAL_DOMAINS.values():
    for domain in domains_list:
        ALL_OFFICIAL_DOMAINS.add(domain.lower())