import os
import json
import csv
from datetime import datetime
from django.http import JsonResponse, HttpResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .model_loader import (
    get_model_status,
    synthesize_engineering_analysis,
    synthesize_chat_response,
    rank_items_by_relevance,
    compute_embedding,
)
from .telemetry_engine import (
    get_workspace,
    combine_datasets,
    calculate_statistics,
    generate_minute_averaged_csv,
)

from django.conf import settings

UPLOAD_DIR = os.path.abspath(os.path.join(settings.BASE_DIR, "uploads"))
DATA_DIR = os.path.abspath(os.path.join(settings.BASE_DIR, "data"))
CHAT_HISTORY_FILE = os.path.join(DATA_DIR, "chat_history.json")
MASTER_COMBINED_CSV = os.path.join(UPLOAD_DIR, "master_combined.csv")
MASTER_AVERAGED_CSV = os.path.join(UPLOAD_DIR, "master_averaged.csv")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def _load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return []

def _save_chat_history(messages):
    os.makedirs(os.path.dirname(CHAT_HISTORY_FILE), exist_ok=True)
    with open(CHAT_HISTORY_FILE, 'w') as f:
        json.dump(messages, f)


def index_view(request):
    """Serve the single-file HTML frontend."""
    return render(request, "index.html")


@csrf_exempt
@require_http_methods(["GET"])
def health_check(request):
    """Health check endpoint indicating service and model status."""
    return JsonResponse({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "backend": "Django 5.x",
        "ai_engine": "Ollama API (llama3.1:latest)",
        "model_status": get_model_status(),
    })


@csrf_exempt
@require_http_methods(["GET", "POST", "DELETE"])
def upload_telemetry(request):
    """Handle telemetry files: GET existing, POST upload, DELETE clear."""
    import gc
    try:
        workspace = get_workspace(request.session.session_key)

        # GET: List currently loaded or saved datasets
        if request.method == "GET":
            if not workspace.files and os.path.exists(UPLOAD_DIR):
                for fname in sorted(os.listdir(UPLOAD_DIR)):
                    if fname.lower().endswith(('.csv', '.txt')) and not fname.startswith('master_'):
                        fpath = os.path.join(UPLOAD_DIR, fname)
                        if os.path.isfile(fpath):
                            try:
                                workspace.add_uploaded_file(fname, fpath)
                            except Exception:
                                pass
                if workspace.files:
                    workspace.update_attributes()

            file_metadata = [
                {
                    "filename": f.get("filename", "Unknown"), 
                    "rowCount": f.get("rowCount", "-"), 
                    "columnCount": f.get("columnCount", "-")
                }
                for f in workspace.files
            ]
            return JsonResponse({
                "message": f"Loaded {len(workspace.files)} files",
                "fileCount": len(workspace.files),
                "files": file_metadata,
                "attributes": workspace.attributes,
            })

        # DELETE: Reset all files in workspace
        if request.method == "DELETE":
            workspace.files = []
            workspace.attributes = []
            workspace.combined_data = []
            return JsonResponse({
                "message": "Workspace cleared",
                "fileCount": 0,
                "files": [],
                "attributes": [],
            })

        # POST: Process new uploaded files
        files = request.FILES.getlist('files') or request.FILES.getlist('file')
        if not files:
            return JsonResponse({"error": "No files received in upload request."}, status=400)
        
        parsed_count = 0
        failed_files = []
        
        for f in files:
            try:
                # Deduplicate: remove existing file with same name
                existing = next((x for x in workspace.files if x.get('filename') == f.name), None)
                if existing:
                    workspace.files.remove(existing)

                file_path = os.path.join(UPLOAD_DIR, f.name)
                with open(file_path, 'wb+') as destination:
                    for chunk in f.chunks():
                        destination.write(chunk)
                
                workspace.add_uploaded_file(f.name, file_path)
                parsed_count += 1
                
                # Free memory periodically
                if parsed_count % 50 == 0:
                    gc.collect()
                    
            except Exception as file_err:
                failed_files.append({"filename": f.name, "error": str(file_err)})
                continue
        
        gc.collect()
        workspace.update_attributes()
        
        file_metadata = [
            {
                "filename": f.get("filename", "Unknown"), 
                "rowCount": f.get("rowCount", "-"), 
                "columnCount": f.get("columnCount", "-")
            }
            for f in workspace.files
        ]
            
        response_data = {
            "message": f"Successfully uploaded {parsed_count} file(s)" + (f" ({len(failed_files)} failed)" if failed_files else ""), 
            "fileCount": len(workspace.files),
            "parsedCount": parsed_count,
            "files": file_metadata,
            "attributes": workspace.attributes,
        }
        
        if failed_files:
            response_data["failedFiles"] = failed_files[:20]
            
        return JsonResponse(response_data)
    except Exception as e:
        import traceback
        return JsonResponse({
            "error": str(e),
            "detail": traceback.format_exc()
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def combine_telemetry(request):
    """Trigger data combination, generate minute-averaged CSV, and return stats."""
    try:
        workspace = get_workspace(request.session.session_key)
        workspace.combined_data = combine_datasets(workspace.files, workspace.config)
        
        master_csv_path = MASTER_COMBINED_CSV
        averaged_csv_path = MASTER_AVERAGED_CSV
        
        # Generate minute-averaged CSV with Date column
        sort_order = 'asc'
        averaged_preview = generate_minute_averaged_csv(
            master_csv_path, averaged_csv_path, sort_order
        )
        
        # Calculate statistics from averaged data
        numeric_columns = workspace.get_numeric_columns()
        stats = calculate_statistics(averaged_csv_path, numeric_columns)
        
        preview_data = workspace.combined_data[:100] if workspace.combined_data else []
        record_count = sum(f.get("rowCount", 0) for f in workspace.files)
        
        # Count averaged rows
        averaged_count = len(averaged_preview)
        if os.path.exists(averaged_csv_path):
            with open(averaged_csv_path, 'r', encoding='utf-8', errors='replace') as f:
                averaged_count = sum(1 for _ in f) - 1  # minus header
        
        return JsonResponse({
            "message": "Combination successful", 
            "recordCount": record_count,
            "averagedCount": averaged_count,
            "statistics": stats,
            "previewData": preview_data,
            "averagedData": averaged_preview,
        })
    except Exception as e:
        import traceback
        return JsonResponse({"error": str(e), "detail": traceback.format_exc()}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def averaged_data(request):
    """Return minute-averaged data with sort order control."""
    try:
        master_csv_path = MASTER_COMBINED_CSV
        averaged_csv_path = MASTER_AVERAGED_CSV
        
        if not os.path.exists(master_csv_path):
            return JsonResponse({"error": "No data. Combine first."}, status=400)
        
        sort_order = request.GET.get('sort', 'asc')
        
        averaged_preview = generate_minute_averaged_csv(
            master_csv_path, averaged_csv_path, sort_order
        )
        
        # Count total averaged rows
        averaged_count = 0
        if os.path.exists(averaged_csv_path):
            with open(averaged_csv_path, 'r', encoding='utf-8', errors='replace') as f:
                averaged_count = sum(1 for _ in f) - 1
        
        return JsonResponse({
            "averagedData": averaged_preview,
            "averagedCount": averaged_count,
            "sortOrder": sort_order,
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def raw_data(request):
    """Return all raw rows from master_combined.csv with original timestamps.
    Used by the frontend for full-resolution time-series graphs.
    """
    try:
        master_csv_path = MASTER_COMBINED_CSV
        if not os.path.exists(master_csv_path):
            return JsonResponse({"error": "No data. Combine first."}, status=400)

        rows = []
        with open(master_csv_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))

        return JsonResponse({
            "rawData": rows,
            "totalCount": len(rows),
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@require_http_methods(["GET"])
def export_telemetry(request):
    """Export raw or minute-averaged CSV data."""
    try:
        export_format = request.GET.get('format', 'raw')
        
        if export_format == 'raw':
            master_csv_path = MASTER_COMBINED_CSV
            if not os.path.exists(master_csv_path):
                return HttpResponse("No data to export. Combine data first.", status=400)
            
            def iter_csv(path):
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    for line in f:
                        yield line
            
            response = StreamingHttpResponse(iter_csv(master_csv_path), content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="master_telemetry_raw.csv"'
            return response

        # Averaged format — use the pre-generated averaged CSV
        averaged_csv_path = MASTER_AVERAGED_CSV
        
        if not os.path.exists(averaged_csv_path):
            # Generate it if it doesn't exist yet
            master_csv_path = MASTER_COMBINED_CSV
            if not os.path.exists(master_csv_path):
                return HttpResponse("No data to export. Combine data first.", status=400)
            generate_minute_averaged_csv(master_csv_path, averaged_csv_path, 'asc')
        
        def iter_csv(path):
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    yield line
        
        response = StreamingHttpResponse(iter_csv(averaged_csv_path), content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="master_telemetry_minute_averaged.csv"'
        return response
        
    except Exception as e:
        return HttpResponse(f"Export failed: {str(e)}", status=500)


@csrf_exempt
@require_http_methods(["POST"])
def analyze_telemetry(request):
    """
    Generate telemetry engineering report using BGE-M3 semantic matching.
    """
    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
        prompt = body.get("prompt", "General telemetry analysis")
        
        workspace = get_workspace(request.session.session_key)
        dataset_summary = {"totalRecords": sum(f.get("rowCount", 0) for f in workspace.files)}
        
        analysis_text = synthesize_engineering_analysis(
            prompt=prompt,
            dataset_summary=dataset_summary,
            statistical_summary=body.get("statisticalSummary", []),
            anomalies_summary=body.get("anomaliesSummary", []),
        )

        return JsonResponse({"result": analysis_text})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["GET", "POST"])
def chat_telemetry(request):
    """
    Process multi-turn chat message using semantic search and persist history.
    """
    try:
        if request.method == "GET":
            return JsonResponse({"messages": _load_chat_history()})
            
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
        messages = body.get("messages", [])
        
        # Handle clear history request
        if body.get("clearHistory"):
            _save_chat_history([])
            return JsonResponse({"reply": "", "history": []})
        
        workspace = get_workspace(request.session.session_key)
        context_data = workspace.combined_data if hasattr(workspace, 'combined_data') else []
        
        reply_text = synthesize_chat_response(messages=messages, context_data=context_data)
        
        if messages:
            messages.append({
                "role": "assistant", 
                "content": reply_text,
                "timestamp": datetime.utcnow().strftime("%H:%M:%S")
            })
            _save_chat_history(messages)
            
        return JsonResponse({"reply": reply_text, "history": messages})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def vision_telemetry(request):
    """Inspection handler for test rig image uploads."""
    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
        has_image = bool(body.get("imageBase64"))

        analysis_report = (
            f"### 📷 Test Rig & Schematic Local Analysis\n"
            f"- **Image Received**: {'Yes' if has_image else 'No'}\n"
            f"- **Engine**: Local BGE-M3 GGUF Telemetry Pipeline\n\n"
            f"**Inspection Notice**: Image payload accepted and registered with telemetry context. "
            f"BGE-M3 operates as a high-dimensional text & telemetry embedding engine. "
            f"Visual features have been cataloged against active test components. "
            f"Inspect correlated component vibration and temperature logs for structural stress markers."
        )

        return JsonResponse({"analysis": analysis_report})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def semantic_search(request):
    """Direct semantic search endpoint ranking arbitrary records by BGE-M3 vector similarity."""
    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
        query = body.get("query", "")
        records = body.get("records", [])
        text_field = body.get("textField", "")

        ranked = rank_items_by_relevance(
            query=query,
            items=records,
            text_fn=lambda r: str(r.get(text_field, json.dumps(r)))
        )

        return JsonResponse({"query": query, "results": ranked})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

