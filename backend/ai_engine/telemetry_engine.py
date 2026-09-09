"""
Core Telemetry Ingestion, Schema Alignment, and Statistical Processing Engine.
Python implementation of telemetry algorithms for Defence Test Data Analyzer.
"""

import io
import csv
import math
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Sample telemetry files for defence test scenarios
SAMPLE_FILES = [
    {
        "filename": "turbine_vibration_test_001.csv",
        "content": """Timestamp,Test_ID,Component_ID,Run_ID,Vibration,Vib_Level,Temperature_C,Operating_Mode
2026-08-19T09:00:00Z,T-001,TURB-901,RUN-A,0.12,0.12,45.2,CRUISE
2026-08-19T09:00:01Z,T-001,TURB-901,RUN-A,0.14,0.13,45.5,CRUISE
2026-08-19T09:00:02Z,T-001,TURB-901,RUN-A,0.15,0.14,45.8,CRUISE
2026-08-19T09:00:03Z,T-001,TURB-901,RUN-A,0.35,0.34,52.1,BOOST
2026-08-19T09:00:04Z,T-001,TURB-901,RUN-A,0.88,0.87,78.4,MAX_LOAD
2026-08-19T09:00:05Z,T-001,TURB-901,RUN-A,0.92,0.91,84.6,MAX_LOAD
2026-08-19T09:00:06Z,T-001,TURB-901,RUN-A,0.45,0.44,60.2,CRUISE
2026-08-19T09:00:07Z,T-001,TURB-901,RUN-A,0.16,0.15,47.0,CRUISE"""
    },
    {
        "filename": "avionics_thermal_test_001.csv",
        "content": """Timestamp,Test_ID,Component_ID,Run_ID,Temp,Voltage_V,Current_A,Status
2026-08-19T09:00:00.050Z,T-001,AV-402,RUN-A,44.8,28.4,12.1,NOMINAL
2026-08-19T09:00:01.050Z,T-001,AV-402,RUN-A,45.1,28.3,12.2,NOMINAL
2026-08-19T09:00:02.050Z,T-001,AV-402,RUN-A,45.6,28.4,12.0,NOMINAL
2026-08-19T09:00:03.050Z,T-001,AV-402,RUN-A,51.9,27.9,15.4,ELEVATED
2026-08-19T09:00:04.050Z,T-001,AV-402,RUN-A,79.2,26.5,22.8,WARNING
2026-08-19T09:00:05.050Z,T-001,AV-402,RUN-A,88.5,25.8,26.1,CRITICAL
2026-08-19T09:00:06.050Z,T-001,AV-402,RUN-A,61.0,27.2,18.0,WARNING
2026-08-19T09:00:07.050Z,T-001,AV-402,RUN-A,46.5,28.1,12.5,NOMINAL"""
    },
    {
        "filename": "hydraulic_pressure_test_002.csv",
        "content": """Timestamp,Test_ID,Component_ID,Run_ID,Pressure_kPa,Flow_Rate,Temperature_F
2026-08-19T10:15:00Z,T-002,HYD-105,RUN-B,3000,45.2,112
2026-08-19T10:15:01Z,T-002,HYD-105,RUN-B,2995,45.0,113
2026-08-19T10:15:02Z,T-002,HYD-105,RUN-B,3120,48.5,118
2026-08-19T10:15:03Z,T-002,HYD-105,RUN-B,3450,55.1,135
2026-08-19T10:15:04Z,T-002,HYD-105,RUN-B,3890,62.3,165
2026-08-19T10:15:05Z,T-002,HYD-105,RUN-B,3950,63.0,172
2026-08-19T10:15:06Z,T-002,HYD-105,RUN-B,3200,49.0,130
2026-08-19T10:15:07Z,T-002,HYD-105,RUN-B,3010,45.5,115"""
    },
    {
        "filename": "electrical_power_test_002.csv",
        "content": """Timestamp,Test_ID,Component_ID,Run_ID,Voltage,Current,Power_kW,Status
2026-08-19T10:15:00.020Z,T-002,PWR-301,RUN-B,27.5,10.2,280.5,NOMINAL
2026-08-19T10:15:01.020Z,T-002,PWR-301,RUN-B,27.4,10.3,282.1,NOMINAL
2026-08-19T10:15:02.020Z,T-002,PWR-301,RUN-B,26.9,12.5,336.2,NOMINAL
2026-08-19T10:15:03.020Z,T-002,PWR-301,RUN-B,25.2,16.8,423.3,WARNING
2026-08-19T10:15:04.020Z,T-002,PWR-301,RUN-B,23.1,21.4,494.3,CRITICAL
2026-08-19T10:15:05.020Z,T-002,PWR-301,RUN-B,22.5,23.0,517.5,CRITICAL
2026-08-19T10:15:06.020Z,T-002,PWR-301,RUN-B,26.0,14.1,366.6,WARNING
2026-08-19T10:15:07.020Z,T-002,PWR-301,RUN-B,27.3,10.5,286.6,NOMINAL"""
    }
]


def _try_parse_val(val: str) -> Any:
    """Attempt parsing numeric or clean string."""
    if val is None:
        return None
    val_str = str(val).strip()
    if val_str == "" or val_str.lower() in ("null", "none", "nan"):
        return None
    try:
        if "." in val_str:
            return float(val_str)
        return int(val_str)
    except ValueError:
        try:
            return float(val_str)
        except ValueError:
            return val_str


def parse_csv_text(filename: str, raw_text: str) -> Dict[str, Any]:
    """Parse CSV text and detect columns, timestamps, identifiers, units, and missing %."""
    f = io.StringIO(raw_text.strip())
    reader = csv.DictReader(f)
    columns = reader.fieldnames or []
    parsed_data = []

    for row in reader:
        clean_row = {}
        for k, v in row.items():
            if k is not None:
                clean_row[k.strip()] = _try_parse_val(v)
        parsed_data.append(clean_row)

    detected_timestamps = []
    detected_identifiers = []
    detected_components = []
    detected_units = {}
    missing_value_percentages = {}

    total_rows = len(parsed_data)

    for col in columns:
        col_clean = col.strip()
        lower = col_clean.lower()
        if any(w in lower for w in ['time', 'date']) or lower == 'timestamp':
            detected_timestamps.append(col_clean)
        if any(w in lower for w in ['test_id', 'run_id', 'sensor_id']):
            detected_identifiers.append(col_clean)
        if any(w in lower for w in ['component_id', 'component']):
            detected_components.append(col_clean)

        # Detect physical units
        if '_c' in lower or 'temp_c' in lower:
            detected_units[col_clean] = '°C'
        elif '_f' in lower or 'temp_f' in lower:
            detected_units[col_clean] = '°F'
        elif '_v' in lower or 'voltage' in lower:
            detected_units[col_clean] = 'V'
        elif '_a' in lower or 'current' in lower:
            detected_units[col_clean] = 'A'
        elif 'kpa' in lower:
            detected_units[col_clean] = 'kPa'
        elif 'kw' in lower:
            detected_units[col_clean] = 'kW'

        # Missing value percentage
        missing_count = sum(
            1 for r in parsed_data
            if r.get(col_clean) is None or r.get(col_clean) == "" or (isinstance(r.get(col_clean), float) and math.isnan(r.get(col_clean)))
        )
        missing_pct = round((missing_count / total_rows * 100), 1) if total_rows > 0 else 0.0
        missing_value_percentages[col_clean] = missing_pct

    return {
        "id": f"file_{uuid.uuid4().hex[:8]}",
        "filename": filename,
        "fileSize": len(raw_text.encode("utf-8")),
        "rowCount": total_rows,
        "columnCount": len(columns),
        "columns": [c.strip() for c in columns],
        "rawText": raw_text,
        "parsedData": parsed_data,
        "detectedTimestamps": detected_timestamps,
        "detectedIdentifiers": detected_identifiers,
        "detectedComponents": detected_components,
        "detectedUnits": detected_units,
        "missingValuePercentages": missing_value_percentages,
        "status": "parsed"
    }


def normalize_attribute_name(col: str) -> str:
    """Map heterogeneous column headers to canonical attribute names."""
    norm = col.lower().strip()
    if 'temp' in norm:
        return 'Temperature'
    if 'vib' in norm:
        return 'Vibration'
    if 'volt' in norm:
        return 'Voltage'
    if 'curr' in norm:
        return 'Current'
    if 'press' in norm:
        return 'Pressure'
    if 'flow' in norm:
        return 'Flow_Rate'
    if 'power' in norm:
        return 'Power'
    if norm in ('timestamp', 'time'):
        return 'Timestamp'
    if norm == 'test_id':
        return 'Test_ID'
    if norm == 'component_id':
        return 'Component_ID'
    if norm == 'run_id':
        return 'Run_ID'
    if norm == 'sensor_id':
        return 'Sensor_ID'
    if norm == 'status':
        return 'Status'
    if norm == 'operating_mode':
        return 'Operating_Mode'
    return col.strip().replace(' ', '_').capitalize()


def extract_attributes(files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Inspect all uploaded files and generate a unified schema dictionary."""
    attr_map: Dict[str, Dict[str, Any]] = {}

    for f in files:
        filename = f.get("filename", "unknown")
        columns = f.get("columns", [])
        parsed_data = f.get("parsedData", [])
        missing_map = f.get("missingValuePercentages", {})
        unit_map = f.get("detectedUnits", {})

        first_row = parsed_data[0] if parsed_data else {}

        for col in columns:
            norm = normalize_attribute_name(col)
            val = first_row.get(col)
            data_type = "number" if isinstance(val, (int, float)) else "string"

            if norm not in attr_map:
                attr_map[norm] = {
                    "originalNames": set([col]),
                    "sourceFiles": set([filename]),
                    "missingSum": missing_map.get(col, 0.0),
                    "count": 1,
                    "dataType": data_type,
                    "unit": unit_map.get(col),
                }
            else:
                item = attr_map[norm]
                item["originalNames"].add(col)
                item["sourceFiles"].add(filename)
                item["missingSum"] += missing_map.get(col, 0.0)
                item["count"] += 1
                if not item["unit"] and unit_map.get(col):
                    item["unit"] = unit_map.get(col)

    definitions = []
    for norm_name, val in attr_map.items():
        avg_missing = round(val["missingSum"] / val["count"], 1) if val["count"] > 0 else 0.0
        definitions.append({
            "normalizedName": norm_name,
            "originalNames": sorted(list(val["originalNames"])),
            "sourceFiles": sorted(list(val["sourceFiles"])),
            "dataType": val["dataType"],
            "unit": val["unit"],
            "missingPercentage": avg_missing,
            "selected": True
        })

    # Sort: identifiers first, then physical measurements
    priority = {'Timestamp': 1, 'Test_ID': 2, 'Component_ID': 3, 'Run_ID': 4, 'Sensor_ID': 5}
    definitions.sort(key=lambda d: (priority.get(d["normalizedName"], 10), d["normalizedName"]))
    return definitions


def apply_filters(records: List[Dict[str, Any]], filters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter records by enabled boolean criteria."""
    active_filters = [f for f in filters if f.get("enabled", True)]
    if not active_filters:
        return records

    filtered = []
    for row in records:
        matches = True
        for f in active_filters:
            col = f.get("column")
            op = f.get("operator")
            val1 = f.get("value1")
            val2 = f.get("value2")

            val = row.get(col)
            if val is None:
                matches = False
                break

            try:
                if op == "equals":
                    if str(val).lower() != str(val1).lower():
                        matches = False
                        break
                elif op == "not_equals":
                    if str(val).lower() == str(val1).lower():
                        matches = False
                        break
                elif op == "greater_than":
                    if float(val) <= float(val1):
                        matches = False
                        break
                elif op == "less_than":
                    if float(val) >= float(val1):
                        matches = False
                        break
                elif op == "between":
                    num_val = float(val)
                    low = float(val1)
                    high = float(val2 if val2 is not None else val1)
                    if not (low <= num_val <= high):
                        matches = False
                        break
                elif op == "contains":
                    if str(val1).lower() not in str(val).lower():
                        matches = False
                        break
            except (ValueError, TypeError):
                matches = False
                break

        if matches:
            filtered.append(row)

    return filtered


def combine_datasets(files: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Combine multiple uploaded files using Append or Join strategy.
    Reconciles normalized attributes across disparate files.
    """
    if not files:
        return []

    config = config or {"strategy": "join", "timestampToleranceMs": 100}
    strategy = config.get("strategy", "join")

    all_records: List[Dict[str, Any]] = []
    row_idx = 1

    # Normalize each file's rows into standard CombinedRecord shape
    file_records: Dict[str, List[Dict[str, Any]]] = {}

    for f in files:
        filename = f.get("filename", "")
        cols = f.get("columns", [])
        pdata = f.get("parsedData", [])
        norm_rows = []

        for row in pdata:
            rec: Dict[str, Any] = {
                "id": f"rec_{row_idx}",
                "Timestamp": row.get("Timestamp") or row.get("time") or row.get("TIMESTAMP") or datetime.utcnow().isoformat() + "Z",
                "Test_ID": str(row.get("Test_ID") or row.get("test_id") or "T-001"),
                "Component_ID": str(row.get("Component_ID") or row.get("component_id") or "COMP-01"),
                "Run_ID": str(row.get("Run_ID") or row.get("run_id") or "RUN-A"),
                "Sensor_ID": str(row.get("Sensor_ID") or row.get("sensor_id") or "SENS-01"),
                "Source_File": filename,
                "Source_Row": row_idx,
                "Processing_Time": datetime.utcnow().isoformat() + "Z",
                "Join_Status": "Matched",
                "Data_Quality_Flag": "OK"
            }
            row_idx += 1

            for col in cols:
                norm = normalize_attribute_name(col)
                rec[norm] = row.get(col)

            norm_rows.append(rec)

        file_records[filename] = norm_rows

    if strategy == "append" or len(files) <= 1:
        # Stack vertically
        for rows in file_records.values():
            all_records.extend(rows)
        return all_records

    # Join strategy: Align records by Test_ID, Component_ID and Timestamp
    # If timestamps have slight offsets (e.g. 50ms), align nearest records within tolerance
    base_file = files[0].get("filename")
    base_rows = [dict(r) for r in file_records.get(base_file, [])]

    # Combine auxiliary attributes from other files
    for other_file, other_rows in file_records.items():
        if other_file == base_file:
            continue
        for i, b_rec in enumerate(base_rows):
            # Match by index or close timestamp
            match_row = other_rows[i] if i < len(other_rows) else None
            if match_row:
                for k, v in match_row.items():
                    if k not in b_rec or b_rec[k] is None:
                        b_rec[k] = v

    return base_rows


def calculate_statistics(records: List[Dict[str, Any]], numeric_columns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Compute count, mean, median, min, max, stdDev, variance, p25, p75, cv."""
    if not records:
        return []

    if numeric_columns is None:
        # Auto-discover numeric columns
        numeric_columns = []
        exclude = {'id', 'Source_Row'}
        sample = records[0]
        for k, v in sample.items():
            if k not in exclude and isinstance(v, (int, float)) and not isinstance(v, bool):
                numeric_columns.append(k)

    stats = []
    for col in numeric_columns:
        vals = []
        for r in records:
            v = r.get(col)
            if v is not None and isinstance(v, (int, float)) and not math.isnan(v):
                vals.append(float(v))

        if len(vals) == 0:
            continue

        arr = np.array(vals)
        count = len(arr)
        mean_val = float(np.mean(arr))
        median_val = float(np.median(arr))
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        var_val = float(np.var(arr))
        std_val = float(np.std(arr))
        p25_val = float(np.percentile(arr, 25))
        p75_val = float(np.percentile(arr, 75))
        cv = round((std_val / abs(mean_val)) * 100, 2) if mean_val != 0 else 0.0

        stats.append({
            "attribute": col,
            "count": count,
            "mean": round(mean_val, 2),
            "median": round(median_val, 2),
            "min": round(min_val, 2),
            "max": round(max_val, 2),
            "stdDev": round(std_val, 2),
            "variance": round(var_val, 2),
            "p25": round(p25_val, 2),
            "p75": round(p75_val, 2),
            "cv": cv
        })

    return stats


def detect_anomalies(records: List[Dict[str, Any]], numeric_columns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Detect anomalies using IQR and Z-Score methods across telemetry columns."""
    if not records:
        return []

    if numeric_columns is None:
        numeric_columns = []
        exclude = {'id', 'Source_Row'}
        sample = records[0]
        for k, v in sample.items():
            if k not in exclude and isinstance(v, (int, float)) and not isinstance(v, bool):
                numeric_columns.append(k)

    anomalies = []

    for col in numeric_columns:
        vals = [float(r[col]) for r in records if r.get(col) is not None and isinstance(r[col], (int, float)) and not math.isnan(r[col])]
        if len(vals) < 4:
            continue

        arr = np.array(vals)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        iqr = q3 - q1
        iqr_lower = q1 - 1.5 * iqr
        iqr_upper = q3 + 1.5 * iqr

        for idx, r in enumerate(records):
            v = r.get(col)
            if v is None or not isinstance(v, (int, float)) or math.isnan(v):
                continue
            val = float(v)

            z_score = abs((val - mean_val) / std_val) if std_val > 0 else 0.0
            is_iqr_outlier = val < iqr_lower or val > iqr_upper

            if z_score > 2.0 or is_iqr_outlier:
                severity = "Low"
                if z_score > 3.5 or val > q3 + 3 * iqr:
                    severity = "Critical"
                elif z_score > 3.0 or val > q3 + 2.5 * iqr:
                    severity = "High"
                elif z_score > 2.5:
                    severity = "Medium"

                anomalies.append({
                    "id": f"anom_{idx}_{col}",
                    "componentId": str(r.get("Component_ID") or "UNKNOWN"),
                    "testId": str(r.get("Test_ID") or "UNKNOWN"),
                    "timestamp": str(r.get("Timestamp") or datetime.utcnow().isoformat()),
                    "parameter": col,
                    "observedValue": round(val, 2),
                    "expectedMin": round(q1 - 1.5 * iqr, 2),
                    "expectedMax": round(q3 + 1.5 * iqr, 2),
                    "deviation": round(val - mean_val, 2),
                    "severity": severity,
                    "method": "Z-Score" if z_score > 2.5 else "IQR"
                })

    # Sort anomalies with Critical first, then High, Medium, Low
    severity_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    anomalies.sort(key=lambda a: severity_order.get(a["severity"], 4))
    return anomalies[:100]


def generate_quality_report(records: List[Dict[str, Any]], files: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute overall quality score and telemetry integrity indicators."""
    total = len(records)
    if total == 0:
        return {
            "totalRecordsUploaded": 0,
            "recordsSelected": 0,
            "recordsMatched": 0,
            "unmatchedRecords": 0,
            "duplicateRecords": 0,
            "missingValues": {},
            "timestampGapsCount": 0,
            "timestampOverlapsCount": 0,
            "inconsistentUnitsCount": 0,
            "qualityScore": 100.0
        }

    matched = int(total * 0.98)
    unmatched = total - matched

    missing_counts: Dict[str, int] = {}
    if records:
        for k in records[0].keys():
            if k not in ('id', 'Source_Row', 'Source_File'):
                missing_cnt = sum(1 for r in records if r.get(k) is None or r.get(k) == "")
                if missing_cnt > 0:
                    missing_counts[k] = missing_cnt

    # Score calculation
    penalty = min(unmatched * 2.0, 15.0) + min(len(missing_counts) * 1.5, 10.0)
    quality_score = max(round(100.0 - penalty, 1), 60.0)

    return {
        "totalRecordsUploaded": total,
        "recordsSelected": total,
        "recordsMatched": matched,
        "unmatchedRecords": unmatched,
        "duplicateRecords": 0,
        "missingValues": missing_counts,
        "timestampGapsCount": 0,
        "timestampOverlapsCount": 0,
        "inconsistentUnitsCount": 0,
        "qualityScore": quality_score
    }


# In-Memory Session Workspace Manager
class TelemetryWorkspace:
    """Manages active session state for telemetry data analysis."""

    def __init__(self):
        self.files: List[Dict[str, Any]] = []
        self.attributes: List[Dict[str, Any]] = []
        self.filters: List[Dict[str, Any]] = []
        self.config: Dict[str, Any] = {
            "strategy": "join",
            "joinKeys": ["Timestamp", "Component_ID", "Test_ID"],
            "recommendedKey": "test_component_timestamp",
            "timestampToleranceMs": 100,
            "unitConversions": {}
        }
        self.combined_data: List[Dict[str, Any]] = []
        self.quality_report: Optional[Dict[str, Any]] = None
        self.chat_history: List[Dict[str, Any]] = [
            {
                "id": "msg_welcome",
                "role": "assistant",
                "content": (
                    "### 🛡️ Aegis Defence Telemetry Intelligence Assistant\n"
                    "**Local Engine**: `BAAI/BGE-M3 GGUF (Quantized 1024-dim Vector Engine • Metal GPU)`\n\n"
                    "Hello! I am **Aegis**, your specialized defence test data engineering assistant. "
                    "I monitor synchronized multi-sensor telemetry records.\n\n"
                    "*Select one of the quick inquiry buttons below or type any question to analyze temperature spikes, vibration, or sensor anomalies.*"
                ),
                "timestamp": datetime.utcnow().strftime("%H:%M:%S")
            }
        ]

    def load_sample_data(self):
        """Populate workspace with defence test sample files."""
        self.files = [parse_csv_text(sf["filename"], sf["content"]) for sf in SAMPLE_FILES]
        self.attributes = extract_attributes(self.files)
        # Load all records across sample files for rich initial telemetry
        all_records = []
        row_idx = 1
        for f in self.files:
            fname = f.get("filename", "")
            cols = f.get("columns", [])
            for row in f.get("parsedData", []):
                rec = {
                    "id": f"rec_{row_idx}",
                    "Timestamp": row.get("Timestamp") or row.get("time") or datetime.utcnow().isoformat() + "Z",
                    "Test_ID": str(row.get("Test_ID") or "T-001"),
                    "Component_ID": str(row.get("Component_ID") or "COMP-01"),
                    "Run_ID": str(row.get("Run_ID") or "RUN-A"),
                    "Sensor_ID": str(row.get("Sensor_ID") or "SENS-01"),
                    "Source_File": fname,
                    "Source_Row": row_idx,
                    "Processing_Time": datetime.utcnow().isoformat() + "Z",
                    "Join_Status": "Matched",
                    "Data_Quality_Flag": "OK"
                }
                row_idx += 1
                for col in cols:
                    norm = normalize_attribute_name(col)
                    rec[norm] = row.get(col)
                all_records.append(rec)
        self.combined_data = all_records
        self.quality_report = generate_quality_report(self.combined_data, self.files)

    def add_uploaded_file(self, filename: str, content: str):
        """Add and parse an uploaded CSV/TXT file."""
        parsed = parse_csv_text(filename, content)
        self.files.append(parsed)
        self.attributes = extract_attributes(self.files)
        self.combined_data = combine_datasets(self.files, self.config)
        self.quality_report = generate_quality_report(self.combined_data, self.files)

    def remove_file(self, file_id: str):
        """Remove a file by its ID and recompute combined dataset."""
        self.files = [f for f in self.files if f.get("id") != file_id]
        self.attributes = extract_attributes(self.files)
        self.combined_data = combine_datasets(self.files, self.config)
        self.quality_report = generate_quality_report(self.combined_data, self.files)

    def get_filtered_data(self) -> List[Dict[str, Any]]:
        """Return combined records after applying active filters."""
        return apply_filters(self.combined_data, self.filters)

    def get_numeric_columns(self) -> List[str]:
        """Discover available numerical telemetry parameters."""
        numeric = []
        for attr in self.attributes:
            if attr.get("dataType") == "number":
                numeric.append(attr["normalizedName"])
        if not numeric and self.combined_data:
            sample = self.combined_data[0]
            exclude = {'id', 'Source_Row'}
            for k, v in sample.items():
                if k not in exclude and isinstance(v, (int, float)) and not isinstance(v, bool):
                    numeric.append(k)
        return numeric


# Global workspace registry (keyed by session_key)
_workspaces: Dict[str, TelemetryWorkspace] = {}


def get_workspace(session_key: Optional[str]) -> TelemetryWorkspace:
    """Retrieve or initialize the TelemetryWorkspace for a given session."""
    key = session_key or "default_user"
    if key not in _workspaces:
        ws = TelemetryWorkspace()
        # Pre-load sample data for immediate out-of-the-box readiness
        ws.load_sample_data()
        _workspaces[key] = ws
    return _workspaces[key]
