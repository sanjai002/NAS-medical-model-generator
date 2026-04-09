"""
app.py  –  Flask Web Application (Backend Server)
====================================================

This is the main entry point of the AutoNAS system.  It creates a Flask
web server that exposes:

  PAGES:
    GET  /              → Training page (index.html) – upload dataset, configure NAS, watch training live.
    GET  /predict_page  → Prediction page (predict.html) – submit patient data, get model prediction.

  API ENDPOINTS:
    POST /upload             → Receive a dataset file + NAS settings, launch NAS in a background thread.
    GET  /stream             → Server-Sent Events (SSE) endpoint – streams live training updates to the browser.
    GET  /prediction_schema  → Returns feature metadata so the prediction form can be built dynamically.
    POST /predict            → Accept patient data JSON/form, preprocess it, run inference, return prediction.
    GET  /download_best      → Download the best trained .keras model file.
    GET  /download_report    → Download the training_report.json file.
    GET  /download_candidate/<id> → Download a specific candidate's .keras model.

  ARCHITECTURE:
    - Training runs in a background daemon thread (_train_worker) so the
      Flask server stays responsive during long NAS runs.
    - A thread-safe queue (event_q) connects _train_worker ↔ SSE stream.
    - RuntimeState dataclass holds the current run's results, protected
      by a threading.Lock (state_lock) for thread-safe reads/writes.
    - The best model is cached in memory (_cached_model) to avoid loading
      it from disk on every prediction request.

  FLOW:
    1. User uploads CSV → POST /upload.
    2. _train_worker starts: loads data, runs NAS, pushes SSE events.
    3. Browser listens on GET /stream, updates Chart.js graphs in real time.
    4. After training, user goes to /predict_page.
    5. Frontend fetches /prediction_schema to build the input form.
    6. User fills form → POST /predict → model inference → JSON result.
"""

# ---------------------------------------------------------------------------
# Environment variable setup (must happen BEFORE importing TensorFlow)
# ---------------------------------------------------------------------------
import os

# Suppress TensorFlow's verbose C++ logging (INFO + WARNING messages).
# Level "2" keeps only ERROR messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Force TensorFlow to use CPU only (no GPU).  This avoids CUDA errors
# on machines without a GPU and keeps the web server lightweight.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------

# threading – runs NAS training in a background thread so Flask stays responsive.
import threading

# queue – thread-safe FIFO queue for passing SSE events from the training
#   thread to the /stream endpoint.
import queue

# json – serializes/deserializes event payloads and the training report.
import json

# dataclass, field – define the RuntimeState container with default factories.
from dataclasses import dataclass, field

# datetime – generates timestamps for filenames and the training report.
from datetime import datetime

# Path – object-oriented file path manipulation (used in _cleanup_old_artifacts).
from pathlib import Path

# Any – generic type hint for dicts with mixed value types.
from typing import Any

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------

# Flask framework components:
#   Flask           – the application object.
#   render_template – renders Jinja2 HTML templates.
#   request         – access incoming HTTP request data (files, form fields, JSON).
#   jsonify         – convert a Python dict to a JSON HTTP response.
#   Response        – used for the SSE streaming response.
#   send_file       – serve a file for download.
from flask import Flask, render_template, request, jsonify, Response, send_file

# secure_filename – sanitizes user-uploaded filenames to prevent path traversal attacks.
from werkzeug.utils import secure_filename

# NumPy – used in the /predict route to process model predictions (argmax, etc.).
import numpy as np

# Pandas – used in the /predict route to build a DataFrame from user input.
import pandas as pd

# TensorFlow – used to load saved .keras models for inference in /predict.
import tensorflow as tf

# Joblib – serializes/deserializes the sklearn preprocessing pipeline
#   and label encoder so /predict can reuse the exact training transforms.
import joblib

# ---------------------------------------------------------------------------
# Local module imports
# ---------------------------------------------------------------------------

# load_and_prepare_dataset – the data pipeline function from data_pipeline.py.
# Takes a file path, returns a PreparedData object ready for training.
from data_pipeline import load_and_prepare_dataset

# NAS search functions and result type from nas_engine.py:
#   CandidateResult        – dataclass returned by each trained candidate.
#   run_evolutionary_search – genetic algorithm search.
#   run_progressive_search  – incremental depth-growing search.
#   run_random_search       – N independent random architectures.
from nas_engine import (
    CandidateResult,
    run_evolutionary_search,
    run_progressive_search,
    run_random_search,
)

# ---------------------------------------------------------------------------
# Flask app configuration
# ---------------------------------------------------------------------------

# Directory where uploaded datasets, trained models, pipeline, and report are saved.
UPLOAD_FOLDER = "uploads"

# Create the uploads directory if it doesn't exist yet.
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create the Flask application instance.
app = Flask(__name__)

# Store the upload folder path in Flask's config so it can be accessed
# anywhere via app.config["UPLOAD_FOLDER"].
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Thread-safe FIFO queue for Server-Sent Events.
# _train_worker puts JSON payloads into this queue, and sse_stream()
# reads from it to yield events to the browser.
event_q = queue.Queue()


# ---------------------------------------------------------------------------
# RuntimeState – in-memory state of the current/last NAS run
# ---------------------------------------------------------------------------

@dataclass
class RuntimeState:
    """
    Holds all mutable state for the current (or most recent) NAS run.

    All fields are read/written under state_lock to ensure thread safety
    between the Flask request threads and the background training thread.

    Attributes
    ----------
    trained_candidates : dict[int, dict[str, Any]]
        Maps candidate_id → metadata dict (architecture, metrics, file path).
        Populated after training completes.
    best_candidate_id : int | None
        ID of the best-performing candidate (highest val_performance).
    best_model_path : str | None
        File path to the best candidate's .keras model.
    report_path : str | None
        File path to training_report.json.
    pipeline_path : str | None
        File path to best_pipeline.joblib (the sklearn preprocessing pipeline).
    feature_schema : list[dict[str, Any]]
        Metadata for each feature column (used by /prediction_schema).
    feature_columns : list[str]
        Original feature column names from the dataset.
    task : str | None
        "classification" or "regression" (set after data is loaded).
    target_name : str | None
        Name of the target column in the dataset.
    run_in_progress : bool
        True while a NAS run is actively training.  The /upload route
        checks this to reject concurrent requests (returns 409).
    """
    trained_candidates: dict[int, dict[str, Any]] = field(default_factory=dict)
    best_candidate_id: int | None = None
    best_model_path: str | None = None
    report_path: str | None = None
    pipeline_path: str | None = None
    feature_schema: list[dict[str, Any]] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    task: str | None = None
    target_name: str | None = None
    run_in_progress: bool = False


# Single global instance of RuntimeState – shared across all threads.
state = RuntimeState()

# Lock that protects reads/writes to the `state` object.
# Must be acquired (via `with state_lock:`) before accessing any state field.
state_lock = threading.Lock()

# Lock that protects loading/accessing the cached Keras model.
model_cache_lock = threading.Lock()

# Module-level model cache – avoids reloading the .keras file on every /predict.
# _cached_model     : the loaded tf.keras.Model object (or None).
# _cached_model_path: the path that was loaded (compared to detect staleness).
_cached_model = None
_cached_model_path = None


# ---------------------------------------------------------------------------
# SSE (Server-Sent Events) helpers
# ---------------------------------------------------------------------------

def push_event(payload, event="update"):
    """
    Serialize a payload dict and push it onto the SSE event queue.

    This function is called from both the main thread (status messages)
    and the background training thread (training metrics).  The queue
    is thread-safe, so no lock is needed.

    Parameters
    ----------
    payload : dict
        Arbitrary data to send to the browser.  Will be JSON-serialized.
    event : str
        SSE event type name.  The browser uses addEventListener(event, ...)
        to route different event types to different handlers.
        Common values: "status", "model_info", "step", "epoch_end",
                       "result", "done", "error".
    """
    # Wrap the payload and event type into one JSON string for the queue.
    event_q.put(json.dumps({"event": event, "data": payload}))


def sse_stream():
    """
    Generator that yields SSE-formatted messages from the event queue.

    This runs inside a Flask streaming Response.  It blocks on
    event_q.get() until a new event is available, then formats it
    according to the SSE protocol:

        event: <event_type>\n
        data: <json_payload>\n\n

    The double newline signals the end of one SSE message.

    Yields
    ------
    str
        SSE-formatted event lines.
    """
    while True:
        # Block until a message is available in the queue.
        msg_json = event_q.get()

        # Parse the wrapper to extract event type and data.
        msg = json.loads(msg_json)

        # Yield SSE-formatted lines to the browser.
        yield f"event: {msg['event']}\n"
        yield f"data: {json.dumps(msg['data'])}\n\n"


@app.route("/stream")
def stream():
    """
    SSE endpoint.  The browser connects to this URL with EventSource('/stream')
    and receives a continuous stream of training events.

    Returns
    -------
    Response
        A Flask streaming response with MIME type "text/event-stream".
    """
    return Response(sse_stream(), mimetype="text/event-stream")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _clamp(value: int, low: int, high: int) -> int:
    """
    Restrict an integer value to the range [low, high].

    Used to constrain user-provided form values (candidates, batch_size,
    epochs) to safe operating ranges.

    Parameters
    ----------
    value : int
        The value to clamp.
    low : int
        Minimum allowed value.
    high : int
        Maximum allowed value.

    Returns
    -------
    int
        The clamped value: max(low, min(high, value)).
    """
    return max(low, min(high, int(value)))


def _parse_int(value: Any, default: int) -> int:
    """
    Safely parse a value to int, returning `default` on failure.

    Handles None, empty strings, non-numeric strings without crashing.

    Parameters
    ----------
    value : Any
        The raw form value (could be str, None, etc.).
    default : int
        Fallback value if parsing fails.

    Returns
    -------
    int
        Parsed integer or the default.
    """
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _cleanup_old_artifacts(exclude: list[Path] | None = None) -> None:
    """
    Delete old model files and uploaded datasets from prior NAS runs.

    Called at the start of each new NAS run to prevent the uploads/ folder
    from accumulating stale files.  The current run's uploaded dataset
    is excluded from deletion.

    Logic:
      - Scans every file in the uploads/ directory.
      - Skips any file whose path matches an entry in `exclude`.
      - Deletes files that look like old candidate models:
          name starts with "candidate_" and extension is .keras or .h5.
      - Deletes files that look like old uploads:
          extension is .csv/.xlsx/.xls and name starts with a digit
          (timestamped filenames like "20260326_162650_data.csv").

    Parameters
    ----------
    exclude : list[Path] | None
        File paths to keep (typically the just-uploaded dataset).
    """
    uploads = Path(app.config["UPLOAD_FOLDER"])
    exclude = exclude or []

    # Iterate over all files in the uploads directory.
    for p in uploads.glob("*"):
        # Skip files that are in the exclusion list (the current upload).
        # samefile() compares actual filesystem identity, not just string paths.
        if any(p.samefile(x) for x in exclude if x.exists()):
            continue

        if p.is_file():
            # Check if this is an old candidate model file (e.g. candidate_1.keras).
            is_old_model = p.name.startswith("candidate_") and p.suffix.lower() in {".keras", ".h5"}

            # Check if this is an old uploaded dataset (timestamped name).
            is_old_upload = p.suffix.lower() in {".csv", ".xlsx", ".xls"} and p.name[:1].isdigit()

            if is_old_model or is_old_upload:
                try:
                    p.unlink()   # Delete the file.
                except Exception:
                    pass         # Silently ignore deletion failures.


# ---------------------------------------------------------------------------
# Report building helpers
# ---------------------------------------------------------------------------

def _serialize_result(r: CandidateResult) -> dict[str, Any]:
    """
    Convert a CandidateResult dataclass into a plain dict for JSON serialization.

    Used when building the training report JSON file.

    Parameters
    ----------
    r : CandidateResult
        The result object for one trained candidate.

    Returns
    -------
    dict[str, Any]
        Dictionary with all key metrics and metadata.
    """
    return {
        "candidate": r.candidate_id,
        "architecture": r.architecture,
        "final_accuracy": r.final_accuracy,
        "final_mse": r.final_mse,
        "final_loss": r.final_loss,
        "final_metric": r.final_metric,
        "total_params": r.total_params,
        "layer_names": r.layer_names,
        "activations": r.activations,
        "model_path": r.model_path,
    }


def _build_report(results: list[CandidateResult], best: CandidateResult, task: str) -> dict[str, Any]:
    """
    Build the full training report dictionary.

    This dict is written to training_report.json and includes:
      - A UTC timestamp of when the report was generated.
      - The task type (classification or regression).
      - A summary of the best candidate's architecture and metrics.
      - A full list of all candidates with their individual metrics.

    Parameters
    ----------
    results : list[CandidateResult]
        All successfully trained candidates.
    best : CandidateResult
        The single best candidate (highest val_performance).
    task : str
        "classification" or "regression".

    Returns
    -------
    dict[str, Any]
        The complete report ready for JSON serialization.
    """
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "task": task,
        "best_candidate": best.candidate_id,
        "best_summary": {
            "architecture": best.architecture,
            "total_params": best.total_params,
            "layer_names": best.layer_names,
            "activations": best.activations,
            "final_accuracy": best.final_accuracy,
            "final_mse": best.final_mse,
            "final_loss": best.final_loss,
        },
        "candidates": [_serialize_result(x) for x in results],
    }


def _choose_best(results: list[CandidateResult]) -> CandidateResult:
    """
    Select the best candidate from all trained results.

    Uses Python's max() on the val_performance field:
      - For classification: val_performance = validation accuracy (higher is better).
      - For regression: val_performance = -validation_MSE (higher/less negative is better).

    Parameters
    ----------
    results : list[CandidateResult]
        All successfully trained candidates (must be non-empty).

    Returns
    -------
    CandidateResult
        The single best candidate.
    """
    return max(results, key=lambda r: r.val_performance)


# ---------------------------------------------------------------------------
# Background training worker
# ---------------------------------------------------------------------------

def _train_worker(filepath: str, nas_type: str, candidates: int, batch_size: int, epochs: int) -> None:
    """
    Background thread function that runs the full NAS pipeline.

    This function is launched by the /upload route in a daemon thread.
    It executes the following steps:

      1. Reset the global RuntimeState (clear previous run's data).
      2. Clean up old artifacts from the uploads/ directory.
      3. Load and preprocess the uploaded dataset using data_pipeline.
      4. Store feature schema and metadata in RuntimeState.
      5. Run the selected NAS strategy (random / evolutionary / progressive).
      6. Select the best candidate.
      7. Write training_report.json to disk.
      8. Save the sklearn preprocessing pipeline via joblib.
      9. Update RuntimeState with all results and file paths.
     10. Clear the model cache (force fresh load on next /predict).
     11. Push a "done" SSE event with the best model summary.

    Error handling:
      - If any exception occurs, an "error" event is pushed, followed
        by a "done" event with "Run aborted" message.
      - The finally block always sets state.run_in_progress = False.

    Parameters
    ----------
    filepath : str
        Path to the uploaded dataset file.
    nas_type : str
        One of "random", "evolutionary", "progressive".
    candidates : int
        Number of candidates to train (or population size for evolutionary).
    batch_size : int
        Training batch size (16 or 32).
    epochs : int
        Maximum training epochs per candidate (1–5).
    """
    # Reference the global model cache variables so we can clear them.
    global _cached_model, _cached_model_path

    # ── Step 1: Reset RuntimeState for the new run ──
    # Acquire the lock and clear all fields from any previous run.
    with state_lock:
        state.trained_candidates = {}
        state.best_candidate_id = None
        state.best_model_path = None
        state.report_path = None
        state.pipeline_path = None
        state.feature_schema = []
        state.feature_columns = []
        state.task = None
        state.target_name = None
        state.run_in_progress = True   # Flag that a run is now active.

    # ── Step 2: Clean up old files (keep the current upload) ──
    _cleanup_old_artifacts(exclude=[Path(filepath)])

    try:
        # ── Step 3: Load and preprocess the dataset ──
        # This calls data_pipeline.load_and_prepare_dataset() which reads the
        # file, auto-detects the task, builds the preprocessor, splits data,
        # and returns a PreparedData object with all arrays.
        data = load_and_prepare_dataset(filepath, max_rows=20_000)

        # Notify the browser that the dataset has been successfully loaded.
        push_event(
            {
                "message": f"Dataset ready. task={data.task}, samples={len(data.X_train) + len(data.X_val) + len(data.X_test)}"
            },
            "status",
        )

        # ── Step 4: Store feature metadata in RuntimeState ──
        # These are needed later by /prediction_schema and /predict routes.
        with state_lock:
            state.feature_schema = data.feature_schema
            state.feature_columns = data.feature_columns
            state.task = data.task
            state.target_name = data.target_name

        # ── Step 5: Run the selected NAS strategy ──
        uploads = app.config["UPLOAD_FOLDER"]

        if nas_type == "random":
            # Random Search: generate and train `candidates` independent architectures.
            results = run_random_search(
                data=data,
                output_dir=uploads,
                push_event=push_event,
                candidate_count=candidates,
                batch_size=batch_size,
                epochs=epochs,
            )
        elif nas_type == "evolutionary":
            # Evolutionary Search: run a genetic algorithm with `candidates` as
            # population size and 3 generations.
            results = run_evolutionary_search(
                data=data,
                output_dir=uploads,
                push_event=push_event,
                population_size=candidates,
                generations=3,
                batch_size=batch_size,
                epochs=epochs,
            )
        else:
            # Progressive Search: start small and incrementally grow the network.
            results = run_progressive_search(
                data=data,
                output_dir=uploads,
                push_event=push_event,
                candidate_limit=candidates,
                batch_size=batch_size,
                epochs=epochs,
            )

        # ── Handle the case where no candidate trained successfully ──
        if not results:
            push_event({"message": "No valid candidate was trained."}, "error")
            push_event({"message": "Run completed with no model."}, "done")
            return

        # ── Step 6: Select the best candidate ──
        best = _choose_best(results)

        # ── Step 7: Build and save the training report to JSON ──
        report = _build_report(results, best, data.task)
        report_path = os.path.join(uploads, "training_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # ── Step 8: Save the sklearn preprocessing pipeline via joblib ──
        # This bundles the fitted ColumnTransformer, LabelEncoder, feature
        # columns, feature schema, task type, and target name into one file.
        # The /predict route loads this to preprocess new input data.
        pipeline_path = os.path.join(uploads, "best_pipeline.joblib")
        joblib.dump(
            {
                "preprocessor": data.preprocessor,
                "label_encoder": data.label_encoder,
                "feature_columns": data.feature_columns,
                "feature_schema": data.feature_schema,
                "task": data.task,
                "target_name": data.target_name,
            },
            pipeline_path,
        )

        # ── Step 9: Update RuntimeState with all results ──
        with state_lock:
            # Store all candidate results as a dict keyed by candidate_id.
            state.trained_candidates = {
                r.candidate_id: {
                    "arch": r.architecture,
                    "final_accuracy": r.final_accuracy,
                    "final_mse": r.final_mse,
                    "final_loss": r.final_loss,
                    "total_params": r.total_params,
                    "layer_names": r.layer_names,
                    "activations": r.activations,
                    "file": r.model_path,
                }
                for r in results
            }
            state.best_candidate_id = best.candidate_id
            state.best_model_path = best.model_path
            state.report_path = report_path
            state.pipeline_path = pipeline_path

        # ── Step 10: Clear the model cache ──
        # A new best model was just trained, so the old cached model is stale.
        # Clear it so the next /predict request loads the fresh model.
        with model_cache_lock:
            if _cached_model is not None:
                del _cached_model          # Free memory held by the old model.
                _cached_model = None
            _cached_model_path = None      # Reset the cached path.
            tf.keras.backend.clear_session()  # Free TF graph memory.

        # ── Step 11: Push the "done" SSE event with best model summary ──
        push_event(
            {
                "message": "NAS completed",
                "best_candidate": best.candidate_id,
                "best_model": {
                    "total_params": best.total_params,
                    "layer_names": best.layer_names,
                    "activations": best.activations,
                    "final_accuracy": best.final_accuracy,
                    "final_mse": best.final_mse,
                    "final_loss": best.final_loss,
                },
            },
            "done",
        )

    except Exception as e:
        # If anything fails, notify the browser with the error message.
        push_event({"message": str(e)}, "error")
        push_event({"message": "Run aborted"}, "done")

    finally:
        # Always mark the run as finished so that new runs can be started.
        with state_lock:
            state.run_in_progress = False


# ===========================================================================
# Flask Routes
# ===========================================================================

# ---------------------------------------------------------------------------
# Page routes (serve HTML templates)
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """
    Serve the main training page (index.html).

    This page contains:
      - A file upload form (dataset + NAS settings).
      - Real-time training charts (connected to /stream SSE).
      - Download buttons for the best model, report, and individual candidates.
    """
    return render_template("index.html")


@app.route("/predict_page")
def predict_page():
    """
    Serve the prediction page (predict.html).

    This page fetches /prediction_schema to build a dynamic input form
    matching the trained model's features, then submits to /predict.
    """
    return render_template("predict.html")


# ---------------------------------------------------------------------------
# /upload – receive dataset and launch NAS training
# ---------------------------------------------------------------------------

@app.route("/upload", methods=["POST"])
def upload():
    """
    Handle dataset upload and start a NAS run in a background thread.

    Expected multipart/form-data fields:
      file       – the CSV or Excel dataset file (required).
      nas_type   – "random", "evolutionary", or "progressive" (default: "random").
      candidates – number of candidates / population size (1–6, default: 4).
      batch_size – training batch size (16–32, default: 32).
      epochs     – max training epochs per candidate (1–5, default: 3).

    Returns
    -------
    JSON
        On success: {"status": "started", "nas_type": ..., "candidates": ..., ...}
        On error:   {"error": "<message>"} with appropriate HTTP status code.
            400 = missing file / unsupported format / invalid NAS type
            409 = another run is already in progress
    """
    # ── Validate the uploaded file ──
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file uploaded"}), 400

    # Only accept CSV and Excel files.
    ext = Path(f.filename).suffix.lower()
    if ext not in {".csv", ".xlsx", ".xls"}:
        return jsonify({"error": "Only CSV and Excel files are supported"}), 400

    # ── Save the file with a timestamp prefix for uniqueness ──
    # secure_filename sanitizes the name to prevent path traversal attacks.
    filename = secure_filename(f.filename)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{stamp}_{filename}")
    f.save(filepath)

    # ── Parse and validate NAS settings from the form ──
    nas_type = request.form.get("nas_type", "random").strip().lower()
    if nas_type not in {"random", "evolutionary", "progressive"}:
        return jsonify({"error": "Invalid NAS type"}), 400

    # Clamp user-provided integers to safe ranges.
    candidates = _clamp(_parse_int(request.form.get("candidates"), 4), 1, 6)   # 1–6
    batch_size = _clamp(_parse_int(request.form.get("batch_size"), 32), 16, 32) # 16–32
    epochs = _clamp(_parse_int(request.form.get("epochs"), 3), 1, 5)           # 1–5

    # ── Reject if a run is already in progress (concurrent runs not supported) ──
    with state_lock:
        if state.run_in_progress:
            return jsonify({"error": "A training run is already in progress"}), 409

    # ── Notify the browser and launch the background training thread ──
    push_event({"message": f"Starting {nas_type} NAS..."}, "status")

    # daemon=True means the thread will be killed when the main process exits.
    threading.Thread(
        target=_train_worker,
        args=(filepath, nas_type, candidates, batch_size, epochs),
        daemon=True
    ).start()

    # Return the confirmed settings to the client.
    return jsonify(
        {
            "status": "started",
            "nas_type": nas_type,
            "candidates": candidates,
            "batch_size": batch_size,
            "epochs": epochs,
        }
    )


# ---------------------------------------------------------------------------
# Download routes – serve trained models and reports
# ---------------------------------------------------------------------------

@app.route("/download_best")
def download_best():
    """
    Download the best candidate's .keras model file.

    Returns 400 JSON error if no model has been trained yet.
    """
    with state_lock:
        if not state.best_model_path:
            return jsonify({"error": "No model found"}), 400
        path = state.best_model_path
    # send_file serves the file as an attachment (triggers browser download).
    return send_file(path, as_attachment=True)


@app.route("/download_report")
def download_report():
    """
    Download the training_report.json file.

    Returns 400 JSON error if no report has been generated yet.
    """
    with state_lock:
        if not state.report_path:
            return jsonify({"error": "No report found"}), 400
        path = state.report_path
    return send_file(path, as_attachment=True)


@app.route("/download_candidate/<int:cid>")
def download_candidate(cid):
    """
    Download a specific candidate's .keras model by its candidate ID.

    Parameters
    ----------
    cid : int
        The candidate ID (from the URL path, e.g. /download_candidate/2).

    Returns 404 if the candidate ID doesn't exist in the current run.
    """
    with state_lock:
        # Look up candidate metadata by ID.
        meta = state.trained_candidates.get(cid)
        if not meta:
            return jsonify({"error": "Not found"}), 404
        path = meta["file"]   # Path to the .keras file.
    return send_file(path, as_attachment=True)


# ---------------------------------------------------------------------------
# /prediction_schema – return feature metadata for the prediction form
# ---------------------------------------------------------------------------

@app.route("/prediction_schema")
def prediction_schema():
    """
    Return JSON metadata describing the trained model's input features.

    The predict.html page calls this endpoint to dynamically build the
    input form.  Each field entry includes the feature name, type
    (numeric / categorical), and allowed values for categoricals.

    Returns
    -------
    JSON
        { "ready": true, "task": ..., "target_name": ...,
          "best_candidate_id": ..., "fields": [...] }
        or
        { "ready": false, "message": "Model is not ready yet" }
    """
    with state_lock:
        # If any essential piece is missing, the model isn't ready.
        if not state.feature_schema or not state.pipeline_path or not state.best_model_path:
            return jsonify({"ready": False, "message": "Model is not ready yet"})

        return jsonify(
            {
                "ready": True,
                "task": state.task,
                "target_name": state.target_name,
                "best_candidate_id": state.best_candidate_id,
                "fields": state.feature_schema,
            }
        )


# ---------------------------------------------------------------------------
# Model caching helper
# ---------------------------------------------------------------------------

def _get_cached_model(model_path: str):
    """
    Load a Keras model from disk, caching it for subsequent calls.

    Thread-safe: uses model_cache_lock.  If the requested path differs
    from the currently cached path, the old model is freed and the new
    one is loaded.

    Parameters
    ----------
    model_path : str
        Path to the .keras model file.

    Returns
    -------
    tf.keras.Model
        The loaded (and cached) Keras model ready for inference.
    """
    global _cached_model, _cached_model_path

    with model_cache_lock:
        # Check if we need to (re)load: either no model cached, or path changed.
        if _cached_model is None or _cached_model_path != model_path:
            if _cached_model is not None:
                del _cached_model                   # Free old model memory.
                tf.keras.backend.clear_session()    # Free TF graph memory.

            # Load the new model from disk.
            _cached_model = tf.keras.models.load_model(model_path)
            _cached_model_path = model_path

        return _cached_model


# ---------------------------------------------------------------------------
# /predict – run inference on user-provided data
# ---------------------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept feature values, preprocess them, run model inference, return result.

    Accepts either JSON body or form-data.  The field names must match the
    feature_columns from the training dataset.

    Processing steps:
      1. Load the preprocessing pipeline from best_pipeline.joblib.
      2. Extract feature values from the request payload.
      3. Build a single-row DataFrame with the correct column order.
      4. Transform it using the fitted sklearn ColumnTransformer.
      5. Run model.predict() on the transformed array.
      6. For classification: return predicted class label + probabilities.
         For regression: return the predicted numeric value.

    Returns
    -------
    JSON
        Classification: {"task": "classification", "prediction": "<label>",
                         "probabilities": {"class_a": 0.8, "class_b": 0.2}}
        Regression:     {"task": "regression", "prediction": 42.7}
        Error:          {"error": "No trained model available"} with 400 status.
    """
    # ── Step 1: Retrieve file paths from state (thread-safe) ──
    with state_lock:
        pipeline_path = state.pipeline_path
        model_path = state.best_model_path

    if not pipeline_path or not model_path:
        return jsonify({"error": "No trained model available"}), 400

    # ── Step 2: Load the saved preprocessing pipeline ──
    # This contains: preprocessor, label_encoder, feature_columns,
    #                feature_schema, task type, and target_name.
    artifacts = joblib.load(pipeline_path)
    preprocessor = artifacts["preprocessor"]              # Fitted ColumnTransformer.
    label_encoder = artifacts.get("label_encoder")        # LabelEncoder (classification only).
    feature_columns = artifacts["feature_columns"]        # Ordered list of feature column names.
    feature_schema = artifacts["feature_schema"]          # Type metadata per feature.
    task = artifacts["task"]                              # "classification" or "regression".

    # ── Step 3: Extract feature values from the request ──
    # Accept both JSON body and form-data.
    payload = request.get_json(silent=True) or request.form.to_dict(flat=True)

    # Build a single row dict, converting each value to the correct type.
    row = {}
    for field in feature_schema:
        name = field["name"]
        raw_val = payload.get(name, None)

        if field["type"] == "numeric":
            # Numeric features: convert to float, or NaN if missing.
            if raw_val in (None, ""):
                row[name] = np.nan
            else:
                row[name] = float(raw_val)
        else:
            # Categorical features: keep as string, default to empty string.
            row[name] = "" if raw_val is None else str(raw_val)

    # ── Step 4: Create a DataFrame and transform it ──
    # The column order must match what the preprocessor was fit on.
    df = pd.DataFrame([row], columns=feature_columns)

    # Apply the same preprocessing (scaling, one-hot encoding) used during training.
    # Result is a dense float32 NumPy array with shape (1, n_transformed_features).
    X = preprocessor.transform(df).astype(np.float32)

    # ── Step 5: Run model inference ──
    model = _get_cached_model(model_path)
    preds = model.predict(X, verbose=0)    # Shape: (1, num_classes) or (1, 1).

    # ── Step 6: Format and return the result ──
    if task == "classification":
        # Get the index of the highest probability.
        idx = int(np.argmax(preds[0]))

        # Convert the numeric index back to the original class label.
        pred_label = label_encoder.inverse_transform([idx])[0] if label_encoder is not None else str(idx)

        # Build a probability map: {"class_name": probability, ...}
        prob_map = {}
        if label_encoder is not None:
            for i, cls in enumerate(label_encoder.classes_):
                prob_map[str(cls)] = float(preds[0][i])

        return jsonify(
            {
                "task": task,
                "prediction": str(pred_label),
                "probabilities": prob_map,
            }
        )

    # Regression: return the raw predicted value.
    return jsonify({"task": task, "prediction": float(preds[0][0])})


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Start the Flask development server.
    #   host="0.0.0.0"        – listen on all network interfaces.
    #   port=5000              – default Flask port.
    #   debug=True             – enable auto-reload and debug error pages.
    #   threaded=True          – handle requests in threads (needed for SSE).
    #   use_reloader=False     – disable file-watcher reloader (avoids double-start
    #                            issues with the background training thread).
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True, use_reloader=False)
