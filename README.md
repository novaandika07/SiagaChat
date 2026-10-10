# 🛡️ SiagaChat

**A scam-link checker that explains its verdict. Built for people in Indonesia, usable anywhere.**

ForgeHacks 2026 · Track 05: AI + Cybersecurity

Paste a suspicious message or link. SiagaChat finds every link in the text, scores how risky the **link text** looks (0–100), tells you **why** in plain words, and tells you **what to do next**. The interface works in English and Bahasa Indonesia.

> **Privacy and safety:** SiagaChat never opens, downloads or contacts the links it checks. It analyses only the text of the URL.

---

## The problem

Scam links reach people through chat apps every day: fake parcel receipts, fake prize notices, fake bank "verification" pages, loan apps, malicious APK files. In Indonesia these often imitate familiar brands (BCA, Shopee, JNE, DANA) and use cheap domains or URL shorteners. Most people cannot tell `shopee.co.id` from `shopee-hadiah-2026.top` at a glance, and a wrong click can cost them their savings.

## What it does

1. **Extracts links** from a pasted message (with or without `http://`, including IP addresses).
2. **Scores each link** from 0 to 100 and labels it **Safe** (< 30), **Caution** (30–69) or **Danger** (≥ 70).
3. **Explains the verdict** with reasons such as *"The domain ending .top is often used for cheap scam sites"* or *"The name 'shopee' is used on a domain that is not official"*.
4. **Gives action steps** that match the label (do not click, never share OTP/PIN, contact the bank through an official channel, how to report in Indonesia).
5. **Switches between English and Bahasa Indonesia**, including the reasons.

## How it works

SiagaChat is a hybrid of a machine-learning model and transparent rules.

```
message → find links → ML score ─┐
                                 ├→ rule adjustments → label + reasons + advice
        brand / domain rules ────┘
```

**1. Machine-learning model.** A Random Forest (compared against Logistic Regression, the better F1 is kept) learns patterns from **17 host-level features**: host and domain length, number of subdomains, hyphens and digits, name randomness (entropy, consonant runs), IP address, URL shortener, cheap TLD, suspicious words, `@`, punycode and port.

**2. Rule layer** (`src/config.py`, `src/predict.py`). The model learns general patterns from public data, but it cannot know local brands, so rules adjust its score:

| Rule | Effect on score |
|---|---|
| Official domain of a known brand | capped at 10 |
| `.go.id` / `.mil.id` | capped at 20 |
| Popular domain (Tranco top 10k) and its own subdomains, but not shared hosting such as `*.blogspot.com` | capped at 20 |
| Brand name imitated on a non-official domain | at least 85 |
| Brand in the path plus a suspicious word | at least 45 |
| IP address as host | at least 55 |
| URL shortener | at least 40 |

**3. Explainable reasons.** Reasons are stored as language-neutral codes (`src/messages.py`) and rendered in English or Indonesian, so switching language does not re-run the model.

**Data.** Kaggle *Malicious URLs dataset* (`malicious_phish.csv`; `defacement` rows and duplicate URLs removed; phishing and malware = malicious) plus the top 50,000 domains of the [Tranco list](https://tranco-list.eu) as benign examples.

**Honest evaluation.** Train and test sets are split **by registered domain**, so the same website never appears on both sides. Scores below therefore measure performance on domains the model has never seen. All numbers come from `python -m src.train` and are saved in `reports/metrics.json`.

## Results (held-out domains)

Test set: 107,325 URLs, 21.0% malicious.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Random Forest (selected)** | 0.708 | 0.350 | 0.453 | **0.395** |
| Logistic Regression (baseline) | 0.721 | 0.358 | 0.414 | 0.384 |

Random Forest confusion matrix: 65,767 safe links correctly passed, 18,993 safe links wrongly flagged, 12,334 scams missed, 10,231 scams caught.

**How to read this honestly:** the ML model alone is weak. It catches about 45% of malicious URLs, and about 65% of its alerts are false alarms. Because 79% of the test set is benign, always answering "benign" would reach 79.0% accuracy, which is higher than our model's 70.8%. We looked only at the **domain name**, not the page path or content, and the Kaggle labels often depend on information we deliberately do not use. We report these numbers as they are rather than hide them.

## What works and what doesn't

**Works (tested by running the app):**
- End-to-end flow: paste text → links found → score, reasons, advice → English/Indonesian switch.
- Rule layer: official domains, popular domains and their subdomains, `.go.id`, imitation of listed brands, shorteners, IP hosts.
- Safe by design: no link is ever opened.

**Does not work / known limits:**
- **The ML score alone is not reliable** (see results). Treat verdicts as hints, not proof. The rule layer carries much of the practical value, and we have **not** measured the rules on a separate test set. We only checked them manually on a small set of examples.
- **Only link text is analysed.** The message wording ("you won a prize!") is not analysed yet, and page content is never checked.
- **False alarms on legitimate but unpopular sites** are likely, especially sites with subdomains.
- **Brand rules focus on Indonesian brands** (`src/config.py`). Imitation of global brands (PayPal, DHL, ...) relies on the ML score unless the brand is added to the list.
- **No LLM runs inside the app.** The verdict comes from the ML model and rules only.
- Example scam links in the app are fictional.

## Run it

Requires Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Option A: use the pre-trained model in models/ (if included) and start the app
python -m streamlit run app/streamlit_app.py

# Option B: retrain from scratch
#   1. put malicious_phish.csv (Kaggle) and tranco.csv (tranco-list.eu) in data/
python -m src.prepare_data
python -m src.train
python -m streamlit run app/streamlit_app.py
```

Command-line checker: `python -m src.predict`

## Project structure

```
app/streamlit_app.py     Streamlit UI (English / Bahasa Indonesia)
src/config.py            brands, official domains, shorteners, TLDs, suspicious words
src/features.py          URL parsing and the 17 features
src/messages.py          reasons in English and Indonesian
src/prepare_data.py      cleaning, feature extraction, split by domain
src/train.py             trains Random Forest and Logistic Regression, saves metrics
src/predict.py           ML score + rule layer → label, reasons, advice
reports/metrics.json     metrics from the training run
```

## Use of AI

We used AI assistants (Claude, ChatGPT and Qwen) for brainstorming, drafting and debugging code, and drafting this README. We ran the code ourselves, trained the model locally, and every number in this README comes from our own training run (`reports/metrics.json`). The scam-detection model is a scikit-learn Random Forest trained on public data. No LLM is called while the app runs.

## Next steps

- Analyse the **message text** (urgency, requests for OTP/PIN, fake relatives), not just links.
- Evaluate the rule layer on a small, hand-labelled set of Indonesian scam and official links.
- Add path-level and richer features, and more Indonesian training data.
- LLM-written plain-language explanations that can never change the score.

## Credits

Kaggle *Malicious URLs dataset*, the Tranco list, scikit-learn, tldextract, RapidFuzz and Streamlit.

Built during ForgeHacks 2026 by Nova Andika.
