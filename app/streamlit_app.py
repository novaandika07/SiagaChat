"""
SiagaChat - Streamlit UI (Stage 1: basic app).

Run from the project root:
    streamlit run app/streamlit_app.py

Safety rule: URLs are NEVER opened, downloaded or fetched.
We only analyze the text of the URL.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import streamlit as st

# Streamlit puts the "app/" folder on sys.path, not the project root.
# Add the root so that "from src.predict import ..." works.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.predict import predict_url  # noqa: E402

MAX_URLS = 5  # analyze at most this many links per message

# ---------------------------------------------------------------------------
# URL extraction
# ---------------------------------------------------------------------------

# Only these bare-domain endings are accepted when there is no http(s):// or
# www. Without this limit, "Halo.Segera" would be mistaken for a domain.
BARE_TLDS = (
    r"com|net|org|id|co\.id|or\.id|go\.id|ac\.id|sch\.id|web\.id|my\.id|"
    r"xyz|top|click|loan|work|tk|ml|ga|cf|gq|win|vip|icu|cfd|sbs|buzz|"
    r"rest|monster|info|online|site|link|app|apk|me|co|io|cc|ly|gl|is|id"
)

URL_PATTERN = re.compile(
    r"(?:https?://[^\s<>\"']+)"                      # with scheme
    r"|(?:www\.[^\s<>\"']+)"                         # starts with www.
    r"|(?:(?<![@\w.\-])"                             # bare domain, not an email
    r"(?:[a-z0-9\-]+\.)+(?:" + BARE_TLDS + r")"
    r"(?![a-z0-9\-])(?:/[^\s<>\"']*)?)",
    flags=re.IGNORECASE,
)

TRAILING_PUNCTUATION = ".,;:!?)]}'\""


def extract_urls(text: str) -> list[str]:
    """Return unique URLs found in text, in order of appearance."""
    found: list[str] = []
    seen: set[str] = set()
    for match in URL_PATTERN.finditer(text):
        url = match.group(0).rstrip(TRAILING_PUNCTUATION)
        key = url.lower()
        if url and key not in seen:
            seen.add(key)
            found.append(url)
    return found


# ---------------------------------------------------------------------------
# Demo examples (all fictional)
# ---------------------------------------------------------------------------

EXAMPLES = {
    "Contoh aman": "Tagihan bisa dicek lewat https://www.bca.co.id ya.",
    "Contoh penipuan 1": (
        "SELAMAT! Anda menang undian Shopee Rp5.000.000. "
        "Klaim hadiah di shopee-hadiah-2026.top/klaim sebelum 24 jam."
    ),
    "Contoh penipuan 2": (
        "Paket Anda tertahan. Cek resi dan update alamat di "
        "http://resi-jne-paket.click/cek agar tidak dikembalikan."
    ),
}


def load_example(text: str) -> None:
    """Button callback: put the example text into the text box."""
    st.session_state["input_text"] = text


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------

def show_result(result: dict) -> None:
    """Draw one result card: colored label, score, reasons."""
    label = result["label"]
    score = result["risk_score"]

    with st.container(border=True):
        # Show the URL as code so it is NOT clickable.
        st.markdown(f"**Link:** `{result['url']}`")

        message = f"{label} (skor risiko {score:.0f}/100)"
        if label == "Aman":
            st.success(message, icon="✅")
        elif label == "Waspada":
            st.warning(message, icon="⚠️")
        else:
            st.error(message, icon="🚨")

        st.progress(int(max(0, min(100, score))))

        st.markdown("**Alasan:**")
        for reason in result.get("reasons", []):
            st.markdown(f"- {reason}")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="SiagaChat", page_icon="🛡️")

st.title("🛡️ SiagaChat")
st.write(
    "Tempel pesan atau link yang mencurigakan. SiagaChat memeriksa **teks "
    "link-nya saja**. Link tidak pernah dibuka."
)

st.caption("Coba contoh:")
cols = st.columns(len(EXAMPLES))
for col, (name, text) in zip(cols, EXAMPLES.items()):
    col.button(
        name,
        on_click=load_example,
        args=(text,),
        use_container_width=True,
    )

user_text = st.text_area(
    "Pesan atau link",
    key="input_text",
    height=150,
    placeholder="Contoh: Selamat anda menang hadiah, klik bit.ly/xxxx ...",
)

if st.button("Periksa", type="primary"):
    if not user_text.strip():
        st.info("Tempel pesan atau link dulu ya.")
    else:
        urls = extract_urls(user_text)
        if not urls:
            st.info(
                "Tidak ada link yang ditemukan di teks ini. Stage ini baru "
                "memeriksa link, belum isi pesan."
            )
        else:
            if len(urls) > MAX_URLS:
                st.caption(
                    f"Ditemukan {len(urls)} link. Hanya {MAX_URLS} pertama "
                    "yang diperiksa."
                )
            for url in urls[:MAX_URLS]:
                try:
                    show_result(predict_url(url))
                except Exception as exc:  # show a friendly error, keep app alive
                    st.error(f"Gagal memeriksa `{url}`: {exc}")

st.divider()
st.caption(
    "⚠️ Hasil adalah perkiraan, bukan kepastian. "
    "Jika ragu, jangan klik dan hubungi layanan resminya."
)
