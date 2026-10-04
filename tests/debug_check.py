"""
Quick diagnostic: model score vs final score vs fired rules.

Run from the project root:
    python -m src.debug_check

Read it like this:
    model  = what the ML model alone says (0-100)
    final  = after the rule layer
    rules  = which rules changed the score ('-' means none)
"""

from src.predict import _trusted_domains, predict_url

URLS = [
    "python.org",
    "docs.python.org",
    "https://www.python.org",
    "github.com",
    "docs.github.com",
    "en.wikipedia.org",
    "mail.google.com",
    "tokopedia.com",
    "evil.blogspot.com",  # shared hosting: must NOT be trusted
    "bca-verifikasi.xyz",  # fake: should be high
]

print(f"Tranco domains loaded: {len(_trusted_domains()):,}")
if not _trusted_domains():
    print("!! data/tranco.csv is missing or empty: the 'popular domain' rule is OFF")
print()
print(f"{'URL':<28}{'model':>7}{'final':>7}  {'label':<8} rules")
print("-" * 70)

for url in URLS:
    try:
        r = predict_url(url)
        rules = ",".join(r["rules_fired"]) or "-"
        print(
            f"{url:<28}{r['model_score']:>7.1f}{r['risk_score']:>7.1f}  "
            f"{r['label']:<8} {rules}"
        )
    except Exception as exc:
        print(f"{url:<28} ERROR: {exc}")