"""
SiagaChat - Streamlit UI (Stage 2: action advice + EN/ID toggle).

Run from the project root:
    python -m streamlit run app/streamlit_app.py

Safety rule: URLs are NEVER opened, downloaded or fetched.
We only analyze the text of the URL.

Reasons come from predict_url as language-neutral items ("reason_items") and
are rendered here with src.messages.render(), so they follow the language
switch too.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import streamlit as st
import tldextract

# Streamlit puts the "app/" folder on sys.path, not the project root.
# Add the root so that "from src.predict import ..." works.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.messages import render  # noqa: E402
from src.predict import predict_url  # noqa: E402

MAX_URLS = 5  # analyze at most this many links per message

# ---------------------------------------------------------------------------
# URL extraction
# ---------------------------------------------------------------------------

# Checks whether the ending of a bare domain (no http://, no www.) is a real
# top-level domain such as .com, .ai, .io or .co.id. It uses the same offline
# public-suffix snapshot as features.py, so every real TLD works without us
# keeping our own list. "Halo.Segera" is rejected because ".segera" is not a TLD.
_TLD_CHECK = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

# Real TLDs that people mostly meet as file extensions (script.py, notes.md).
FILE_LIKE_TLDS = {"py", "md", "sh", "zip", "mov", "pl", "rs"}

URL_PATTERN = re.compile(
    r"(?:https?://[^\s<>\"']+)"                               # with scheme
    r"|(?:www\.[^\s<>\"']+)"                                  # starts with www.
    r"|(?:(?<![@\w.\-/])\d{1,3}(?:\.\d{1,3}){3}"             # IPv4 address
    r"(?::\d+)?(?:/[^\s<>\"']*)?)"
    r"|(?:(?<![@\w.\-/])(?:[a-z0-9\-]+\.)+[a-z0-9\-]{2,}"    # bare name.tld
    r"(?:/[^\s<>\"']*)?)",
    flags=re.IGNORECASE,
)

IPV4_RE = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")
SINGLE_TOKEN_RE = re.compile(r"[^\s.]+(?:\.[^\s.]+)+/?\S*")

TRAILING_PUNCTUATION = ".,;:!?)]}'\""


def _is_real_bare_domain(candidate: str) -> bool:
    """True if a scheme-less candidate ends in a real TLD (or is an IPv4)."""
    host = candidate.split("/", 1)[0].split(":", 1)[0].lower()
    if IPV4_RE.fullmatch(host):
        return True
    ext = _TLD_CHECK(host)
    if not ext.domain or not ext.suffix:
        return False
    return ext.suffix.split(".")[-1] not in FILE_LIKE_TLDS


def extract_urls(text: str) -> list[str]:
    """Return unique URLs found in text, in order of appearance."""
    found: list[str] = []
    seen: set[str] = set()

    def add(url: str) -> None:
        url = url.rstrip(TRAILING_PUNCTUATION)
        key = url.lower()
        if url and key not in seen:
            seen.add(key)
            found.append(url)

    for match in URL_PATTERN.finditer(text):
        candidate = match.group(0)
        has_scheme = candidate.lower().startswith(("http://", "https://", "www."))
        if has_scheme or _is_real_bare_domain(candidate.rstrip(TRAILING_PUNCTUATION)):
            add(candidate)

    # If the whole input is one word that looks like a link (for example a
    # test domain such as login-verify.example), check it even when its ending
    # is not a registered TLD. A bare ".ai" or "io" has no name, so it stays out.
    if not found:
        token = text.strip()
        if SINGLE_TOKEN_RE.fullmatch(token):
            add(token)

    return found


# ---------------------------------------------------------------------------
# Texts (English / Indonesian)
# ---------------------------------------------------------------------------

LANGS = {"English": "en", "Bahasa Indonesia": "id"}

# predict_url returns these three labels (Indonesian names are internal keys).
LABELS = ("Aman", "Waspada", "Bahaya")

TEXTS = {
    "en": {
        "title": "🛡️ SiagaChat",
        "intro": (
            "Paste a suspicious message or link. SiagaChat checks **only the "
            "text of the link**. It never opens it."
        ),
        "examples_caption": "Try an example:",
        "input_label": "Message or link",
        "placeholder": "Example: You won a prize! Claim it at bit.ly/xxxx ...",
        "check_button": "Check",
        "empty_input": "Paste a message or link first.",
        "no_links": (
            "No links found in this text. Paste a full link such as "
            "paypal-login.xyz/verify, or a message that contains one. This "
            "version only checks links, not the message text."
        ),
        "too_many": "Found {total} links. Only the first {max} are checked.",
        "link_label": "Link",
        "score_line": "{label} (risk score {score:.0f}/100)",
        "labels": {"Aman": "Safe", "Waspada": "Caution", "Bahaya": "Danger"},
        "reasons_title": "Why:",
        "advice_title": "What to do",
        "error_line": "Could not check `{url}`: {error}",
        "disclaimer": (
            "⚠️ Results are estimates, not certainty. If in doubt, do not "
            "click and contact the official service."
        ),
        "examples": [
            ("Safe example", "Docs are here: https://docs.python.org"),
            (
                "Scam example 1",
                "Your parcel is on hold. Pay a $1.99 fee to reschedule "
                "delivery: http://dhl-parcel-reschedule.top/pay",
            ),
            (
                "Scam example 2",
                "Your PayPal account is limited. Verify now to avoid "
                "suspension: https://paypal-secure-login.xyz/verify",
            ),
        ],
        "advice": {
            "Aman": {
                "summary": (
                    "No scam signs were found in the domain name. This is not "
                    "a guarantee, and the page content was not checked."
                ),
                "steps": [
                    "Still check who sent you the link and whether you "
                    "expected it.",
                    "Never share OTP codes, PINs or passwords, even on sites "
                    "that look safe.",
                    "If something feels off, open the official site or app "
                    "yourself instead of using the link.",
                ],
            },
            "Waspada": {
                "summary": (
                    "Some signs look suspicious. Be careful before doing "
                    "anything."
                ),
                "steps": [
                    "Do not click the link yet.",
                    "Do not enter personal data, OTP codes or PINs.",
                    "Open the official site or app yourself (type the address "
                    "or use the app) instead of using the link.",
                    "Ask the sender through a different channel, for example "
                    "a call to a number you already know.",
                    "If it turns out to be a scam, report it to iasc.ojk.go.id "
                    "or OJK Contact 157 (Indonesia).",
                ],
            },
            "Bahaya": {
                "summary": "This link looks like a scam. Do not use it.",
                "steps": [
                    "Do not click the link and do not forward it to others.",
                    "Do not fill in any data, and do not install any file "
                    "(especially APK files) from this link.",
                    "Contact your bank or the service directly through its "
                    "official channel (the app, or the number on your card), "
                    "not through details in the message.",
                    "If you already clicked or entered data: change your "
                    "passwords right away, turn on two-step verification, and "
                    "ask your bank to secure or block the account or card.",
                    "Report it quickly. In Indonesia: iasc.ojk.go.id or OJK "
                    "Contact 157 (24 hours). Funds are more likely to be "
                    "frozen when you report early.",
                ],
            },
        },
    },
    "id": {
        "title": "🛡️ SiagaChat",
        "intro": (
            "Tempel pesan atau link yang mencurigakan. SiagaChat memeriksa "
            "**teks link-nya saja**. Link tidak pernah dibuka."
        ),
        "examples_caption": "Coba contoh:",
        "input_label": "Pesan atau link",
        "placeholder": "Contoh: Selamat anda menang hadiah, klik bit.ly/xxxx ...",
        "check_button": "Periksa",
        "empty_input": "Tempel pesan atau link dulu ya.",
        "no_links": (
            "Tidak ada link yang ditemukan di teks ini. Tempel link "
            "lengkap seperti paypal-login.xyz/verify, atau pesan yang "
            "memuat link. Versi ini baru memeriksa link, belum isi pesan."
        ),
        "too_many": (
            "Ditemukan {total} link. Hanya {max} pertama yang diperiksa."
        ),
        "link_label": "Link",
        "score_line": "{label} (skor risiko {score:.0f}/100)",
        "labels": {"Aman": "Aman", "Waspada": "Waspada", "Bahaya": "Bahaya"},
        "reasons_title": "Alasan:",
        "advice_title": "Yang harus dilakukan",
        "error_line": "Gagal memeriksa `{url}`: {error}",
        "disclaimer": (
            "⚠️ Hasil adalah perkiraan, bukan kepastian. Jika ragu, jangan "
            "klik dan hubungi layanan resminya."
        ),
        "examples": [
            ("Contoh aman", "Tagihan bisa dicek lewat https://www.bca.co.id ya."),
            (
                "Contoh penipuan 1",
                "SELAMAT! Anda menang undian Shopee Rp5.000.000. Klaim "
                "hadiah di shopee-hadiah-2026.top/klaim sebelum 24 jam.",
            ),
            (
                "Contoh penipuan 2",
                "Paket Anda tertahan. Cek resi dan update alamat di "
                "http://resi-jne-paket.click/cek agar tidak dikembalikan.",
            ),
        ],
        "advice": {
            "Aman": {
                "summary": (
                    "Tidak ada ciri penipuan yang terdeteksi pada nama "
                    "domain. Ini bukan jaminan, dan isi halaman tidak "
                    "diperiksa."
                ),
                "steps": [
                    "Tetap periksa siapa pengirim link dan apakah kamu "
                    "memang menunggunya.",
                    "Jangan pernah membagikan kode OTP, PIN, atau password, "
                    "meski di situs yang tampak aman.",
                    "Jika ada yang janggal, buka situs atau aplikasi "
                    "resminya sendiri, jangan lewat link itu.",
                ],
            },
            "Waspada": {
                "summary": (
                    "Ada beberapa tanda yang mencurigakan. Hati-hati sebelum "
                    "melakukan apa pun."
                ),
                "steps": [
                    "Jangan klik link-nya dulu.",
                    "Jangan isi data pribadi, kode OTP, atau PIN.",
                    "Buka situs atau aplikasi resminya sendiri (ketik "
                    "alamatnya atau pakai aplikasi), jangan lewat link itu.",
                    "Tanyakan ke pengirim lewat kanal lain, misalnya telepon "
                    "ke nomor yang sudah kamu kenal.",
                    "Jika ternyata penipuan, laporkan ke iasc.ojk.go.id atau "
                    "Kontak OJK 157.",
                ],
            },
            "Bahaya": {
                "summary": (
                    "Link ini terlihat seperti penipuan. Jangan digunakan."
                ),
                "steps": [
                    "Jangan klik link-nya dan jangan teruskan ke orang lain.",
                    "Jangan isi data apa pun, dan jangan install file "
                    "(terutama file APK) dari link ini.",
                    "Hubungi bank atau layanan resminya lewat kanal resmi "
                    "(aplikasi, atau nomor di kartu kamu), bukan lewat "
                    "kontak yang tertulis di pesan.",
                    "Jika sudah terlanjur klik atau mengisi data: segera "
                    "ganti password, aktifkan verifikasi dua langkah, dan "
                    "minta bank mengamankan atau memblokir rekening atau "
                    "kartu.",
                    "Segera laporkan: iasc.ojk.go.id atau Kontak OJK 157 "
                    "(24 jam). Dana lebih mungkin diblokir jika dilaporkan "
                    "lebih awal.",
                ],
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Callbacks and display helpers
# ---------------------------------------------------------------------------

def load_example(text: str) -> None:
    """Button callback: put the example text in the box, clear old results."""
    st.session_state["input_text"] = text
    st.session_state.pop("outcome", None)


def run_check(text: str) -> dict:
    """
    Analyze the text and return a language-independent 'outcome' dict.
    Results are stored in session_state, so switching language re-draws them
    without running the model again.
    """
    if not text.strip():
        return {"kind": "empty"}

    urls = extract_urls(text)
    if not urls:
        return {"kind": "no_links"}

    results, errors = [], []
    for url in urls[:MAX_URLS]:
        try:
            results.append(predict_url(url))
        except Exception as exc:  # keep the app alive on bad input
            errors.append((url, str(exc)))

    return {"kind": "ok", "total": len(urls), "results": results, "errors": errors}


def show_advice(label: str, t: dict) -> None:
    """Draw the 'what to do' block for a label."""
    advice = t["advice"].get(label)
    if advice is None:
        return
    st.markdown(f"**{t['advice_title']}**")
    st.markdown(advice["summary"])
    steps = "\n".join(f"{i}. {step}" for i, step in enumerate(advice["steps"], 1))
    st.markdown(steps)


def show_result(result: dict, t: dict, lang: str) -> None:
    """Draw one result card: colored label, score, reasons, advice."""
    label = result["label"]
    score = float(result["risk_score"])
    display = t["labels"].get(label, label)

    with st.container(border=True):
        # Show the URL as code so it is NOT clickable.
        safe_url = str(result["url"]).replace("`", "'")
        st.markdown(f"**{t['link_label']}:** `{safe_url}`")

        message = t["score_line"].format(label=display, score=score)
        if label == "Aman":
            st.success(message, icon="✅")
        elif label == "Waspada":
            st.warning(message, icon="⚠️")
        else:
            st.error(message, icon="🚨")

        st.progress(int(max(0, min(100, score))))

        items = result.get("reason_items")
        if items is not None:
            reasons = [render(item, lang) for item in items]
        else:  # older predict.py without reason_items: Indonesian text only
            reasons = result.get("reasons", [])
        if reasons:
            st.markdown(f"**{t['reasons_title']}**")
            for reason in reasons:
                st.markdown(f"- {reason}")

        st.divider()
        show_advice(label, t)


def show_outcome(outcome: dict, t: dict, lang: str) -> None:
    kind = outcome["kind"]
    if kind == "empty":
        st.info(t["empty_input"])
    elif kind == "no_links":
        st.info(t["no_links"])
    else:
        if outcome["total"] > MAX_URLS:
            st.caption(t["too_many"].format(total=outcome["total"], max=MAX_URLS))
        for result in outcome["results"]:
            show_result(result, t, lang)
        for url, error in outcome["errors"]:
            st.error(t["error_line"].format(url=url.replace("`", "'"), error=error))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="SiagaChat", page_icon="🛡️")

# Language switch. The first option (English) is the default.
lang_name = st.sidebar.radio("Language / Bahasa", list(LANGS), key="lang_name")
lang = LANGS[lang_name]
t = TEXTS[lang]

st.title(t["title"])
st.write(t["intro"])

st.caption(t["examples_caption"])
cols = st.columns(len(t["examples"]))
for col, (name, example_text) in zip(cols, t["examples"]):
    col.button(
        name,
        key=f"example_{lang}_{name}",
        on_click=load_example,
        args=(example_text,),
        use_container_width=True,
    )

user_text = st.text_area(
    t["input_label"],
    key="input_text",
    height=150,
    placeholder=t["placeholder"],
)

if st.button(t["check_button"], type="primary"):
    st.session_state["outcome"] = run_check(user_text)

outcome = st.session_state.get("outcome")
if outcome:
    show_outcome(outcome, t, lang)

st.divider()
st.caption(t["disclaimer"])