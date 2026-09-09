# Defence Test Data Analyzer

A high-performance defence telemetry analyzer with a **Python (Django 5)** backend, local **Qwen2.5 Chat** & **BGE-M3** AI engines, and a zero-dependency mission-control frontend (`index.html`).

---

## Prerequisites

- **Python**: `3.10` or `3.11+`
- **RAM**: Minimum 4 GB (8 GB recommended for local LLM inference)

---

## Quick Start

### 1. Setup Virtual Environment & Dependencies

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Start the Server

```bash
python backend/manage.py runserver 8000
```

> **Important**: The server must run on port **`8000`** (`http://127.0.0.1:8000`) for the frontend telemetry API to communicate.

### 3. Open the Application

Open your browser and go to:
```
http://127.0.0.1:8000
```

---

## AI Models (`models/`)

The local AI engine stores quantized GGUF models in the `models/` folder:

| Model | File | Function |
| :--- | :--- | :--- |
| **Qwen2.5-0.5B-Instruct** | `qwen2.5-0.5b-instruct-q4_k_m.gguf` | Interactive AI Chat assistant (Aegis) |
| **BAAI/BGE-M3** | `bge-m3-q4_k_m.gguf` | Semantic telemetry ranking & anomaly retrieval |

*Models download automatically if missing when triggered by the backend.*

---

## Key Endpoints

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/` | `GET` | Serves the Mission Control dashboard (`index.html`) |
| `/api/health` | `GET` | Checks backend and AI model loading status |
| `/api/gemini/chat` | `POST` | Processes interactive chat with Qwen2.5 |
| `/api/gemini/analyze` | `POST` | Generates engineering reports using BGE-M3 |

---

## Troubleshooting

- **Server Port in Use**: If port 8000 is occupied, free it using `lsof -ti :8000 | xargs kill -9` before starting.
- **Model Check**: Verify AI status at `http://127.0.0.1:8000/api/health`.
- **Large Files**: The backend supports file uploads up to 50 MB by default.

