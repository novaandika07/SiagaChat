"""
SiagaChat configuration.

Local (Indonesian) knowledge lives here. The ML model learns generic URL
patterns from public data; this file feeds the rule layer that knows about
Indonesian brands, shorteners and trusted domains.

IMPORTANT: double-check OFFICIAL_DOMAINS before you submit. A missing official
domain makes a real site look like a fake one.
"""

# ------------------------------------------------------------
# Brands that scammers like to imitate.
# Short/ambiguous words (pos, jet, ninja, pin) are left out on purpose,
# because they appear in too many harmless domains.
# ------------------------------------------------------------
BRAND_KEYWORDS = [
    # banks
    "bca", "bri", "bni", "mandiri", "cimb", "btn", "danamon", "permata",
    # e-commerce
    "shopee", "tokopedia", "lazada", "blibli", "bukalapak", "tiktok",
    # courier
    "jne", "jnt", "sicepat", "anteraja",
    # e-wallet / fintech
    "gopay", "gojek", "ovo", "dana", "linkaja", "kredivo", "akulaku",
    # telco
    "telkomsel", "indihome",
    # government
    "bpjs", "dukcapil", "pajak",
]

# ------------------------------------------------------------
# Official domains per brand (lowercase).
# A host is official if it equals one of these or is a subdomain of one.
# ------------------------------------------------------------
OFFICIAL_DOMAINS = {
    "bca": ["bca.co.id", "klikbca.com"],
    "bri": ["bri.co.id"],
    "bni": ["bni.co.id"],
    "mandiri": ["bankmandiri.co.id"],
    "cimb": ["cimbniaga.co.id"],
    "btn": ["btn.co.id"],
    "danamon": ["danamon.co.id"],
    "permata": ["permatabank.com"],
    "shopee": ["shopee.co.id", "shopee.com"],
    "tokopedia": ["tokopedia.com"],
    "lazada": ["lazada.co.id", "lazada.com"],
    "blibli": ["blibli.com"],
    "bukalapak": ["bukalapak.com"],
    "tiktok": ["tiktok.com"],
    "jne": ["jne.co.id"],
    "jnt": ["jet.co.id"],
    "sicepat": ["sicepat.com"],
    "anteraja": ["anteraja.id"],
    "gopay": ["gopay.co.id"],
    "gojek": ["gojek.com"],
    "ovo": ["ovo.id"],
    "dana": ["dana.id"],
    "linkaja": ["linkaja.id"],
    "kredivo": ["kredivo.com"],
    "akulaku": ["akulaku.com"],
    "telkomsel": ["telkomsel.com"],
    "indihome": ["indihome.co.id"],
    "bpjs": ["bpjs-kesehatan.go.id", "bpjsketenagakerjaan.go.id"],
    "dukcapil": ["dukcapil.kemendagri.go.id"],
    "pajak": ["pajak.go.id"],
}

# Flat tuple for fast "is this host official?" checks.
ALL_OFFICIAL_DOMAINS = tuple(
    sorted({d.lower() for domains in OFFICIAL_DOMAINS.values() for d in domains})
)

# ------------------------------------------------------------
# Suffixes that ordinary people/scammers cannot register freely.
# ------------------------------------------------------------
TRUSTED_SUFFIXES = ("go.id", "mil.id")

# ------------------------------------------------------------
# Words common in scam URLs (Indonesian + English).
# Words shorter than 6 letters must match a whole token (login.php, not
# "pinterest" for "pin"). Longer words may match as a substring.
# ("dana" is NOT here: it is a brand, handled by the brand rules.)
# ------------------------------------------------------------
SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "verifikasi", "secure", "update", "confirm",
    "reset", "unlock", "akun", "blokir", "hadiah", "klaim", "gratis",
    "menang", "undian", "resi", "paket", "undangan", "apk", "download",
    "instal", "cair", "pencairan", "ktp", "otp", "pin", "password",
    "bonus", "promo", "kaget", "pinjol",
]

# ------------------------------------------------------------
# URL shorteners (hide the final destination).
# ------------------------------------------------------------
SHORTENERS = [
    "bit.ly", "tinyurl.com", "s.id", "cutt.ly", "short.link", "t.co",
    "ow.ly", "goo.gl", "rebrand.ly", "is.gd", "shorturl.at", "tiny.cc",
    "rb.gy", "t.ly",
]

# ------------------------------------------------------------
# Cheap TLDs often abused by scammers.
# ("info", "online", "site" were removed: too many legitimate sites.)
# ------------------------------------------------------------
SUSPICIOUS_TLDS = [
    "xyz", "top", "click", "loan", "work", "tk", "ml", "ga", "cf", "gq",
    "win", "vip", "icu", "cfd", "sbs", "buzz", "rest", "monster",
]