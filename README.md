# Zero-Shot Dynamic Routing System

Intent classification with runtime label updates — no retraining required.

---

## Overview

A production-ready intent routing system built on `facebook/bart-large-mnli`. It classifies user utterances into intents **without any training data** — labels can be swapped at runtime via a REST API, making it useful for rapidly evolving dialogue systems.

**Key capabilities:**
- Zero-shot classification using BART large MNLI
- Out-of-scope (OOS) detection via a configurable confidence threshold
- Runtime label updates without restarting the server
- REST API (FastAPI) + interactive UI (Streamlit)
- TF-IDF + Logistic Regression baseline for benchmarking

---

## Architecture

```
User Utterance
      │
      ▼
┌─────────────┐     POST /predict      ┌──────────────────────┐
│  Streamlit  │ ──────────────────────▶ │     FastAPI Server   │
│   app.py    │ ◀────────────────────── │     router.py        │
└─────────────┘   intent + confidence  └──────────┬───────────┘
                                                   │
                                        ┌──────────▼───────────┐
                                        │  zeroshotClassifier  │
                                        │   classifier.py      │
                                        │                      │
                                        │  facebook/bart-large │
                                        │       -mnli          │
                                        └──────────┬───────────┘
                                                   │
                                        ┌──────────▼───────────┐
                                        │  Label humanization  │
                                        │  e.g. flight_status  │
                                        │  → "flight status"   │
                                        └──────────────────────┘
```

**Label flow:** Raw snake_case labels (e.g. `flight_status`) are humanized before being passed to BART as candidate labels. The reverse map restores original label names in the response.

**OOS detection:** If the top confidence score falls below `OOS_THRESHOLD`, the utterance is flagged as out-of-scope (`is_oos: true`) regardless of the predicted intent.

---

## Project Structure

```
zero-shot-dynamic-routing/
├── api/
│   ├── main.py          # FastAPI app entry point
│   ├── router.py        # Endpoint definitions
│   └── schemas.py       # Pydantic request/response models
├── app/
│   └── app.py           # Streamlit UI
├── src/
│   ├── classifier.py    # BART zero-shot classifier
│   ├── baseline.py      # TF-IDF + LogReg baseline
│   ├── config.py        # Env config (model, threshold, data path)
│   ├── data_loader.py   # JSON data loader
│   └── evaluator.py     # Accuracy + F1 + OOS metrics
├── notebooks/
│   ├── 1_eda.ipynb
│   └── 2_benchmarking.ipynb
├── data/
│   └── data_full.json
├── .env
└── requirements.txt
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- ~2 GB disk space for the BART model (auto-downloaded on first run)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/zero-shot-dynamic-routing.git
cd zero-shot-dynamic-routing
```

### 2. Create and activate a virtual environment

```bash
python -m venv myvenv
# Windows
myvenv\Scripts\activate
# macOS/Linux
source myvenv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
MODEL_NAME=facebook/bart-large-mnli
OOS_THRESHOLD=0.05
DATA_PATH=./data/data_full.json
```

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `facebook/bart-large-mnli` | HuggingFace model ID |
| `OOS_THRESHOLD` | `0.3` | Confidence below this → out-of-scope |
| `DATA_PATH` | *(required)* | Path to `data_full.json` |

> **Note:** The model is cached at `~/.cache/huggingface/` after the first download.

### 5. Start the API server

```bash
uvicorn api.main:app --reload
```

API will be available at `http://127.0.0.1:8000`  
Interactive docs at `http://127.0.0.1:8000/docs`

### 6. Start the Streamlit UI (separate terminal)

```bash
streamlit run app/app.py
```

---

## API Reference

### `POST /predict`

Classify an utterance against the active label set.

**Request**
```json
{
  "text": "what's the weather like in Mumbai?"
}
```

**Response**
```json
{
  "text": "what's the weather like in Mumbai?",
  "intent": "weather",
  "confidence": 0.8731,
  "is_oos": false
}
```

| Field | Type | Description |
|---|---|---|
| `text` | string | The input utterance (whitespace-stripped) |
| `intent` | string | Predicted intent label |
| `confidence` | float | BART softmax score for the top label |
| `is_oos` | bool | `true` if confidence < `OOS_THRESHOLD` |

---

### `GET /labels`

Return the currently active intent labels.

**Response**
```json
{
  "count": 5,
  "labels": ["flight_status", "transfer", "weather", "timer", "translate"]
}
```

---

### `PUT /update-labels`

Replace the active label set at runtime. No restart or retraining required.

**Request**
```json
{
  "labels": ["book_flight", "cancel_order", "track_package"]
}
```

**Response**
```json
{
  "previous_count": 5,
  "new_count": 3,
  "labels": ["book_flight", "cancel_order", "track_package"]
}
```

---

## License

MIT — see [LICENSE](LICENSE) for details.