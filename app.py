import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import threading
import queue
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, render_template, request, jsonify, Response, send_file
from werkzeug.utils import secure_filename

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from data_pipeline import load_and_prepare_dataset
from nas_engine import (
    CandidateResult,
    run_evolutionary_search,
    run_progressive_search,
    run_random_search,
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

event_q = queue.Queue()


@dataclass
class RuntimeState:
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


state = RuntimeState()
state_lock = threading.Lock()
model_cache_lock = threading.Lock()
_cached_model = None
_cached_model_path = None


# ---------------- SSE ----------------
def push_event(payload, event="update"):
    event_q.put(json.dumps({"event": event, "data": payload}))


def sse_stream():
    while True:
        msg_json = event_q.get()
        msg = json.loads(msg_json)
        yield f"event: {msg['event']}\n"
        yield f"data: {json.dumps(msg['data'])}\n\n"


@app.route("/stream")
def stream():
    return Response(sse_stream(), mimetype="text/event-stream")


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _parse_int(value: Any, default: int) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _cleanup_old_artifacts(exclude: list[Path] | None = None) -> None:
    """Delete previous models/uploads but skip paths in `exclude`."""
    uploads = Path(app.config["UPLOAD_FOLDER"])
    exclude = exclude or []
    # Remove old model artifacts and uploaded datasets from prior runs.
    for p in uploads.glob("*"):
        if any(p.samefile(x) for x in exclude if x.exists()):
            continue
        if p.is_file():
            is_old_model = p.name.startswith("candidate_") and p.suffix.lower() in {".keras", ".h5"}
            is_old_upload = p.suffix.lower() in {".csv", ".xlsx", ".xls"} and p.name[:1].isdigit()
            if is_old_model or is_old_upload:
                try:
                    p.unlink()
                except Exception:
                    pass


def _serialize_result(r: CandidateResult) -> dict[str, Any]:
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
    return max(results, key=lambda r: r.val_performance)


def _train_worker(filepath: str, nas_type: str, candidates: int, batch_size: int, epochs: int) -> None:
    global _cached_model, _cached_model_path

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
        state.run_in_progress = True

    _cleanup_old_artifacts(exclude=[Path(filepath)])

    try:
        data = load_and_prepare_dataset(filepath, max_rows=20_000)
        push_event(
            {
                "message": f"Dataset ready. task={data.task}, samples={len(data.X_train) + len(data.X_val) + len(data.X_test)}"
            },
            "status",
        )

        with state_lock:
            state.feature_schema = data.feature_schema
            state.feature_columns = data.feature_columns
            state.task = data.task
            state.target_name = data.target_name

        uploads = app.config["UPLOAD_FOLDER"]
        if nas_type == "random":
            results = run_random_search(
                data=data,
                output_dir=uploads,
                push_event=push_event,
                candidate_count=candidates,
                batch_size=batch_size,
                epochs=epochs,
            )
        elif nas_type == "evolutionary":
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
            results = run_progressive_search(
                data=data,
                output_dir=uploads,
                push_event=push_event,
                candidate_limit=candidates,
                batch_size=batch_size,
                epochs=epochs,
            )

        if not results:
            push_event({"message": "No valid candidate was trained."}, "error")
            push_event({"message": "Run completed with no model."}, "done")
            return

        best = _choose_best(results)
        report = _build_report(results, best, data.task)

        report_path = os.path.join(uploads, "training_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

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

        with state_lock:
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

        with model_cache_lock:
            if _cached_model is not None:
                del _cached_model
                _cached_model = None
            _cached_model_path = None
            tf.keras.backend.clear_session()

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
        push_event({"message": str(e)}, "error")
        push_event({"message": "Run aborted"}, "done")
    finally:
        with state_lock:
            state.run_in_progress = False


# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict_page")
def predict_page():
    return render_template("predict.html")


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file uploaded"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in {".csv", ".xlsx", ".xls"}:
        return jsonify({"error": "Only CSV and Excel files are supported"}), 400

    filename = secure_filename(f.filename)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{stamp}_{filename}")
    f.save(filepath)

    nas_type = request.form.get("nas_type", "random").strip().lower()
    if nas_type not in {"random", "evolutionary", "progressive"}:
        return jsonify({"error": "Invalid NAS type"}), 400

    candidates = _clamp(_parse_int(request.form.get("candidates"), 4), 1, 6)
    batch_size = _clamp(_parse_int(request.form.get("batch_size"), 32), 16, 32)
    epochs = _clamp(_parse_int(request.form.get("epochs"), 3), 1, 5)

    with state_lock:
        if state.run_in_progress:
            return jsonify({"error": "A training run is already in progress"}), 409

    push_event({"message": f"Starting {nas_type} NAS..."}, "status")

    threading.Thread(
        target=_train_worker,
        args=(filepath, nas_type, candidates, batch_size, epochs),
        daemon=True
    ).start()

    return jsonify(
        {
            "status": "started",
            "nas_type": nas_type,
            "candidates": candidates,
            "batch_size": batch_size,
            "epochs": epochs,
        }
    )


@app.route("/download_best")
def download_best():
    with state_lock:
        if not state.best_model_path:
            return jsonify({"error": "No model found"}), 400
        path = state.best_model_path
    return send_file(path, as_attachment=True)


@app.route("/download_report")
def download_report():
    with state_lock:
        if not state.report_path:
            return jsonify({"error": "No report found"}), 400
        path = state.report_path
    return send_file(path, as_attachment=True)


@app.route("/download_candidate/<int:cid>")
def download_candidate(cid):
    with state_lock:
        meta = state.trained_candidates.get(cid)
        if not meta:
            return jsonify({"error": "Not found"}), 404
        path = meta["file"]
    return send_file(path, as_attachment=True)


@app.route("/prediction_schema")
def prediction_schema():
    with state_lock:
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


def _get_cached_model(model_path: str):
    global _cached_model, _cached_model_path
    with model_cache_lock:
        if _cached_model is None or _cached_model_path != model_path:
            if _cached_model is not None:
                del _cached_model
                tf.keras.backend.clear_session()
            _cached_model = tf.keras.models.load_model(model_path)
            _cached_model_path = model_path
        return _cached_model


@app.route("/predict", methods=["POST"])
def predict():
    with state_lock:
        pipeline_path = state.pipeline_path
        model_path = state.best_model_path
    if not pipeline_path or not model_path:
        return jsonify({"error": "No trained model available"}), 400

    artifacts = joblib.load(pipeline_path)
    preprocessor = artifacts["preprocessor"]
    label_encoder = artifacts.get("label_encoder")
    feature_columns = artifacts["feature_columns"]
    feature_schema = artifacts["feature_schema"]
    task = artifacts["task"]

    payload = request.get_json(silent=True) or request.form.to_dict(flat=True)

    row = {}
    for field in feature_schema:
        name = field["name"]
        raw_val = payload.get(name, None)
        if field["type"] == "numeric":
            if raw_val in (None, ""):
                row[name] = np.nan
            else:
                row[name] = float(raw_val)
        else:
            row[name] = "" if raw_val is None else str(raw_val)

    df = pd.DataFrame([row], columns=feature_columns)
    X = preprocessor.transform(df).astype(np.float32)

    model = _get_cached_model(model_path)
    preds = model.predict(X, verbose=0)

    if task == "classification":
        idx = int(np.argmax(preds[0]))
        pred_label = label_encoder.inverse_transform([idx])[0] if label_encoder is not None else str(idx)
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

    return jsonify({"task": task, "prediction": float(preds[0][0])})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True, use_reloader=False)
