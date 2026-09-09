import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import requests

logger = logging.getLogger(__name__)

# ── Ollama API Configuration ────────────────────────────────────
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://172.16.110.46:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "300"))


# ── Core Ollama API Caller ──────────────────────────────────────

def call_ollama_api(prompt: str, model: str = None) -> Optional[str]:
    """Call Ollama API to generate a response."""
    model = model or OLLAMA_MODEL
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 4000,
            }
        }
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "")
        else:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error calling Ollama API: {e}")
        return None
    except Exception as e:
        logger.error(f"Error calling Ollama API: {e}")
        return None


# ── Model / API Status ──────────────────────────────────────────

def get_model_status() -> Dict[str, Any]:
    """Check connectivity to the Ollama API and return status info."""
    status_info = {
        "status": "unknown",
        "ollama_url": OLLAMA_BASE_URL,
        "model_name": OLLAMA_MODEL,
        "error": None,
    }
    try:
        # Ollama exposes a simple GET endpoint at root for health check
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        if resp.status_code == 200:
            tags = resp.json()
            available_models = [m.get("name", "") for m in tags.get("models", [])]
            status_info["status"] = "ready"
            status_info["available_models"] = available_models
            status_info["model_available"] = any(
                OLLAMA_MODEL in m for m in available_models
            )
        else:
            status_info["status"] = "api_error"
            status_info["error"] = f"HTTP {resp.status_code}"
    except requests.exceptions.RequestException as e:
        status_info["status"] = "unreachable"
        status_info["error"] = str(e)
    return status_info


# ── Pseudo-Embedding Utilities (lightweight, no model needed) ───

_embedding_cache: Dict[str, np.ndarray] = {}


def compute_embedding(text: str) -> np.ndarray:
    """
    Compute a deterministic pseudo-embedding for basic semantic ranking.
    Uses a hash-seeded random vector (no external model required).
    """
    if not text:
        text = "empty"

    cache_key = text.strip()
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    np.random.seed(abs(hash(text)) % (2**32))
    vec = np.random.randn(1024).astype(np.float32)
    normalized_vec = vec / np.linalg.norm(vec)
    _embedding_cache[cache_key] = normalized_vec
    return normalized_vec


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two normalized vectors."""
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return float(dot / norm) if norm > 0 else 0.0


def rank_items_by_relevance(
    query: str,
    items: List[Dict[str, Any]],
    text_fn,
    max_items: int = 5,
) -> List[Dict[str, Any]]:
    """Rank candidate items by pseudo-semantic similarity to query."""
    if not items:
        return []

    candidate_items = items[:max_items]
    query_vec = compute_embedding(query)
    ranked = []
    for item in candidate_items:
        text = text_fn(item)
        item_vec = compute_embedding(text)
        sim = cosine_similarity(query_vec, item_vec)
        ranked.append({**item, "_similarity_score": round(sim, 4)})

    ranked.sort(key=lambda x: x["_similarity_score"], reverse=True)
    return ranked


# ── Engineering Analysis via Ollama ─────────────────────────────

def synthesize_engineering_analysis(
    prompt: str,
    dataset_summary: Dict[str, Any],
    statistical_summary: List[Dict[str, Any]],
    anomalies_summary: List[Dict[str, Any]],
) -> str:
    """
    Generate an engineering telemetry analysis report using the Ollama API.
    Sends all telemetry context as a structured prompt.
    """
    total_records = dataset_summary.get("totalRecords", 0)
    critical_count = sum(
        1 for a in anomalies_summary if a.get("severity") == "Critical"
    )
    high_count = sum(
        1 for a in anomalies_summary if a.get("severity") == "High"
    )

    # Build statistical context
    stats_text = ""
    for s in statistical_summary[:8]:
        stats_text += (
            f"  - {s.get('attribute')}: mean={s.get('mean')}, max={s.get('max')}, "
            f"min={s.get('min')}, stdDev={s.get('stdDev')}, CV={s.get('cv')}%\n"
        )

    # Build anomalies context
    anomalies_text = ""
    for a in anomalies_summary[:10]:
        anomalies_text += (
            f"  - Test {a.get('testId')}, Component {a.get('componentId')}, "
            f"Parameter {a.get('parameter')}: observed={a.get('observedValue')}, "
            f"expected=[{a.get('expectedMin')} - {a.get('expectedMax')}], "
            f"deviation={a.get('deviation')}, severity={a.get('severity')}, "
            f"method={a.get('method')}\n"
        )

    llm_prompt = f"""You are Aegis, a defence test data engineering AI assistant.
Generate a detailed engineering telemetry analysis report based on the data below.

USER QUERY: {prompt}

DATASET OVERVIEW:
- Total records: {total_records}
- Total anomalies: {len(anomalies_summary)} ({critical_count} Critical, {high_count} High)
- Dataset summary: {dataset_summary}

STATISTICAL SUMMARY:
{stats_text if stats_text else '  No statistical data available.'}

ANOMALY DETECTIONS:
{anomalies_text if anomalies_text else '  No anomalies detected.'}

FORMAT YOUR RESPONSE AS A STRUCTURED MARKDOWN REPORT WITH:
1. Executive Telemetry Overview
2. Priority Telemetry Indicators (reference the statistics)
3. Traceable Anomaly Detections (use a markdown table with columns: Test ID, Component ID, Parameter, Observed, Expected Range, Deviation, Severity)
4. Engineering Recommendations & Operational Mitigations

Start with: ### 🛡️ Defence Engineering Telemetry Analysis Report
Include: **Engine**: Ollama llama3.1 API
Include: **Query Focus**: *"{prompt}"*
"""

    result = call_ollama_api(llm_prompt)

    if result:
        return result

    # ── Fallback: build a basic static report if Ollama is unreachable ──
    logger.warning("Ollama API unreachable — generating fallback static report.")
    return _fallback_static_analysis(
        prompt, dataset_summary, statistical_summary, anomalies_summary,
        total_records, critical_count, high_count,
    )


def _fallback_static_analysis(
    prompt, dataset_summary, statistical_summary, anomalies_summary,
    total_records, critical_count, high_count,
) -> str:
    """Static fallback report when Ollama API is unavailable."""
    report_lines = [
        "### 🛡️ Defence Engineering Telemetry Analysis Report",
        "**Engine**: Ollama API (Offline — Fallback Mode)",
        f'**Query Focus**: *"{prompt}"*',
        "",
        "#### 1. Executive Telemetry Overview",
        f"- **Dataset Volume**: {total_records} records with {len(anomalies_summary)} anomalies "
        f"({critical_count} Critical, {high_count} High).",
        "",
        "#### 2. Priority Telemetry Indicators",
    ]

    for stat in statistical_summary[:4]:
        report_lines.append(
            f"- **{stat.get('attribute')}**: "
            f"Mean: `{stat.get('mean')}`, Max: `{stat.get('max')}`, "
            f"Min: `{stat.get('min')}`, StdDev: `{stat.get('stdDev')}`, "
            f"CV: `{stat.get('cv')}%`"
        )

    report_lines.append("\n#### 3. Traceable Anomaly Detections")
    if anomalies_summary:
        report_lines.append(
            "| Test ID | Component ID | Parameter | Observed | Expected Range | Deviation | Severity |"
        )
        report_lines.append(
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
        )
        for a in anomalies_summary[:6]:
            exp_range = f"[{a.get('expectedMin', '—')} - {a.get('expectedMax', '—')}]"
            report_lines.append(
                f"| `{a.get('testId', 'N/A')}` | `{a.get('componentId', 'N/A')}` | "
                f"**{a.get('parameter', 'N/A')}** | `{a.get('observedValue', 'N/A')}` | "
                f"{exp_range} | `{a.get('deviation', 'N/A')}` | **{a.get('severity', 'Low')}** |"
            )
    else:
        report_lines.append("No anomalies detected.")

    report_lines.extend([
        "",
        "#### 4. Engineering Recommendations",
        "1. Schedule non-destructive testing (NDT) on components with critical deviations.",
        "2. Verify sensor baseline calibration where CV exceeds normal operating envelopes.",
        "3. Correlate peak load durations against secondary test logs.",
        "",
        "> ⚠️ *This is a fallback report. Ollama API was unreachable for full AI analysis.*",
    ])

    return "\n".join(report_lines)


# ── Chat via Ollama ─────────────────────────────────────────────

def _build_telemetry_summary(context_data: List[Dict[str, Any]]) -> str:
    """Build a concise telemetry summary string for the LLM system prompt."""
    if not context_data:
        return "No telemetry data is currently loaded."

    components = sorted(
        {str(r.get("Component_ID")) for r in context_data if r.get("Component_ID")}
    )
    tests = sorted(
        {str(r.get("Test_ID")) for r in context_data if r.get("Test_ID")}
    )
    files = sorted(
        {str(r.get("Source_File")) for r in context_data if r.get("Source_File")}
    )

    def _stats(keys):
        vals = []
        for r in context_data:
            for k in keys:
                v = r.get(k)
                if v is not None and isinstance(v, (int, float)):
                    vals.append(float(v))
                    break
        if not vals:
            return None
        return {
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
            "avg": round(sum(vals) / len(vals), 2),
            "count": len(vals),
        }

    temp_s = _stats(["Temperature", "Temperature_C", "Temp"])
    vib_s = _stats(["Vibration", "Vib_Level", "Vib"])
    volt_s = _stats(["Voltage", "Voltage_V", "Volt"])
    press_s = _stats(["Pressure", "Pressure_kPa", "Pressure_PSI"])
    curr_s = _stats(["Current_A", "Current"])
    power_s = _stats(["Power_kW", "Power"])

    critical_count = sum(
        1 for r in context_data
        if str(r.get("Status", "")).upper() in ("CRITICAL", "FAILURE", "FAIL")
    )
    warning_count = sum(
        1 for r in context_data
        if str(r.get("Status", "")).upper() in ("WARNING", "ELEVATED")
    )

    lines = [
        f"Records: {len(context_data)} | Files: {len(files)} | "
        f"Components: {', '.join(components)} | Tests: {', '.join(tests)}",
    ]
    if temp_s:
        lines.append(f"Temperature(°C): min={temp_s['min']}, max={temp_s['max']}, avg={temp_s['avg']}")
    if vib_s:
        lines.append(f"Vibration(g): min={vib_s['min']}, max={vib_s['max']}, avg={vib_s['avg']}")
    if volt_s:
        lines.append(f"Voltage(V): min={volt_s['min']}, max={volt_s['max']}, avg={volt_s['avg']}")
    if press_s:
        lines.append(f"Pressure(kPa): min={press_s['min']}, max={press_s['max']}, avg={press_s['avg']}")
    if curr_s:
        lines.append(f"Current(A): min={curr_s['min']}, max={curr_s['max']}, avg={curr_s['avg']}")
    if power_s:
        lines.append(f"Power(kW): min={power_s['min']}, max={power_s['max']}, avg={power_s['avg']}")
    if critical_count or warning_count:
        lines.append(f"Anomalies: {critical_count} Critical, {warning_count} Warning")

    # Add a few sample rows for grounding
    sample_rows = context_data[:5]
    if sample_rows:
        lines.append("Sample records:")
        for r in sample_rows:
            parts = []
            for k in [
                "Test_ID", "Component_ID", "Timestamp", "Temperature",
                "Vibration", "Voltage", "Pressure", "Current_A",
                "Power_kW", "Status",
            ]:
                v = r.get(k)
                if v is not None:
                    parts.append(f"{k}={v}")
            lines.append("  " + ", ".join(parts))

    return "\n".join(lines)


def synthesize_chat_response(
    messages: List[Dict[str, str]],
    context_data: List[Dict[str, Any]],
) -> str:
    """
    Generate a chat response using the Ollama API.
    Falls back to a simple message if the API is unreachable.
    """
    last_query = (
        messages[-1].get("content", "").strip() if messages else "Summary"
    )

    telemetry_summary = _build_telemetry_summary(context_data)

    # Build conversation history for context
    history_text = ""
    recent = [m for m in messages if m.get("role") in ("user", "assistant")]
    for m in recent[-6:]:
        role_label = "User" if m["role"] == "user" else "Aegis"
        history_text += f"{role_label}: {m['content']}\n"

    llm_prompt = f"""You are Aegis, an AI assistant for defence test data engineering.
You help engineers analyze telemetry data from multi-sensor test runs.
Answer the user's question directly and concisely.
If the question is conversational (greetings, how are you, etc.), respond naturally and briefly.
If the question is about telemetry data, use ONLY the data provided below — never invent numbers.
Keep answers short (2-4 sentences for simple questions, more detail only when explicitly asked).

TELEMETRY DATA:
{telemetry_summary}

CONVERSATION HISTORY:
{history_text}

User: {last_query}
Aegis:"""

    result = call_ollama_api(llm_prompt)

    if result:
        return result

    # ── Fallback when Ollama is unreachable ──
    logger.warning("Ollama API unreachable — using fallback response.")

    if not context_data:
        return "No telemetry records are currently loaded. Please upload test CSV files first."

    # Provide a basic context-aware fallback
    ranked_records = rank_items_by_relevance(
        query=last_query,
        items=context_data,
        text_fn=lambda r: " ".join(
            f"{k}={v}" for k, v in r.items()
            if v is not None and k != "_similarity_score"
        ),
        max_items=5,
    )

    if ranked_records:
        top = ranked_records[0]
        detail_parts = [
            f"**{k}**: `{v}`"
            for k, v in top.items()
            if v is not None
            and k != "_similarity_score"
            and k not in ("Source_Row", "Processing_Time", "Join_Status")
        ]
        return (
            f"⚠️ *Ollama API is currently unreachable. Showing basic semantic search results.*\n\n"
            f'Based on search for *"{last_query}"*, the most relevant record is:\n\n'
            + " | ".join(detail_parts[:8])
            + f"\n\n(Relevance: `{top.get('_similarity_score', 'N/A')}`)"
        )

    return (
        f"I couldn't find relevant telemetry data for *\"{last_query}\"*. "
        f"Try asking about specific components, parameters, or test runs."
    )
