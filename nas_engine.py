from __future__ import annotations

import gc
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import tensorflow as tf

from data_pipeline import PreparedData


UNITS_CHOICES = [16, 32, 64, 128]
ACTIVATIONS = ["relu", "tanh"]
MAX_PARAMS = 1_000_000


@dataclass
class CandidateResult:
    candidate_id: int
    architecture: dict
    model_path: str
    total_params: int
    layer_names: list[str]
    activations: list[str]
    final_loss: float
    final_metric: float
    final_accuracy: float | None
    final_mse: float | None
    val_performance: float



def random_architecture() -> dict:
    n_layers = random.randint(1, 4)
    return {
        "dense_units": [random.choice(UNITS_CHOICES) for _ in range(n_layers)],
        "activation": random.choice(ACTIVATIONS),
    }



def mutate_architecture(arch: dict) -> dict:
    out = {
        "dense_units": list(arch["dense_units"]),
        "activation": arch["activation"],
    }

    mutation = random.choice(["add", "remove", "change"])
    if mutation == "add" and len(out["dense_units"]) < 4:
        out["dense_units"].append(random.choice(UNITS_CHOICES))
    elif mutation == "remove" and len(out["dense_units"]) > 1:
        out["dense_units"].pop(random.randrange(len(out["dense_units"])))
    else:
        idx = random.randrange(len(out["dense_units"]))
        out["dense_units"][idx] = random.choice(UNITS_CHOICES)

    if random.random() < 0.3:
        out["activation"] = random.choice(ACTIVATIONS)

    return out



def grow_architecture(arch: dict) -> dict:
    if len(arch["dense_units"]) >= 4:
        return arch
    out = {
        "dense_units": list(arch["dense_units"]) + [random.choice(UNITS_CHOICES)],
        "activation": random.choice(ACTIVATIONS),
    }
    return out



def build_model(input_dim: int, output_dim: int, task: str, arch: dict) -> tf.keras.Model:
    x_in = tf.keras.Input(shape=(input_dim,))
    x = x_in
    for units in arch["dense_units"]:
        x = tf.keras.layers.Dense(units, activation=arch["activation"])(x)

    if task == "classification":
        y = tf.keras.layers.Dense(output_dim, activation="softmax")(x)
    else:
        y = tf.keras.layers.Dense(1, activation="linear")(x)

    return tf.keras.Model(inputs=x_in, outputs=y)



def _safe_count_params(model: tf.keras.Model) -> int:
    try:
        return int(model.count_params())
    except Exception:
        return MAX_PARAMS + 1



def _metric_name(task: str) -> str:
    return "accuracy" if task == "classification" else "mse"


class StreamCallback(tf.keras.callbacks.Callback):
    def __init__(self, push_event: Callable, candidate_id: int, task: str):
        super().__init__()
        self.push_event = push_event
        self.candidate_id = candidate_id
        self.task = task
        self.batch_index = 0

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.batch_index += 1

        weight_norms = []
        for lyr in self.model.layers:
            if isinstance(lyr, tf.keras.layers.Dense):
                weights = lyr.get_weights()
                kernel_norm = float(np.linalg.norm(weights[0])) if weights else 0.0
                weight_norms.append({"layer": lyr.name, "norm": kernel_norm})

        metric_key = _metric_name(self.task)
        batch_metric = float(logs.get(metric_key, logs.get("loss", 0.0)))

        self.push_event(
            {
                "candidate": self.candidate_id,
                "batch": self.batch_index,
                "loss": float(logs.get("loss", 0.0)),
                "batch_metric": batch_metric,
                "weight_norms": weight_norms,
            },
            "step",
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metric_key = _metric_name(self.task)
        self.push_event(
            {
                "candidate": self.candidate_id,
                "epoch": int(epoch + 1),
                "val_loss": float(logs.get("val_loss", 0.0)),
                "val_metric": float(logs.get(f"val_{metric_key}", 0.0)),
            },
            "epoch_end",
        )



def _compile_model(model: tf.keras.Model, task: str):
    if task == "classification":
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=["mse"],
        )



def _fit_candidate(
    data: PreparedData,
    arch: dict,
    candidate_id: int,
    model_path: str,
    push_event: Callable,
    batch_size: int,
    epochs: int,
) -> CandidateResult | None:
    tf.keras.backend.clear_session()

    model = None
    try:
        model = build_model(data.input_dim, data.output_dim, data.task, arch)
        total_params = _safe_count_params(model)
        if total_params > MAX_PARAMS:
            push_event(
                {
                    "candidate": candidate_id,
                    "message": f"Skipped candidate {candidate_id}: parameters exceed limit",
                    "total_params": total_params,
                },
                "status",
            )
            return None

        _compile_model(model, data.task)

        layer_names = [lyr.name for lyr in model.layers if isinstance(lyr, tf.keras.layers.Dense)]
        layer_acts = [
            getattr(lyr, "activation", None).__name__ if hasattr(getattr(lyr, "activation", None), "__name__") else "linear"
            for lyr in model.layers
            if isinstance(lyr, tf.keras.layers.Dense)
        ]

        push_event(
            {
                "candidate": candidate_id,
                "arch": arch,
                "layer_names": layer_names,
                "layer_activations": layer_acts,
                "total_params": total_params,
            },
            "model_info",
        )

        train_ds = (
            tf.data.Dataset.from_tensor_slices((data.X_train, data.y_train))
            .shuffle(1024)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices((data.X_val, data.y_val))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        cb_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)
        cb_stream = StreamCallback(push_event=push_event, candidate_id=candidate_id, task=data.task)

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=max(1, min(epochs, 5)),
            verbose=0,
            callbacks=[cb_early, cb_stream],
        )

        test_ds = tf.data.Dataset.from_tensor_slices((data.X_test, data.y_test)).batch(batch_size)
        eval_vals = model.evaluate(test_ds, verbose=0)
        val_vals = model.evaluate(val_ds, verbose=0)

        final_loss = float(eval_vals[0])
        final_metric = float(eval_vals[1]) if len(eval_vals) > 1 else float("nan")

        if data.task == "classification":
            final_accuracy = final_metric
            final_mse = None
            val_perf = float(val_vals[1]) if len(val_vals) > 1 else 0.0
        else:
            final_accuracy = None
            final_mse = final_metric
            val_perf = -float(val_vals[1]) if len(val_vals) > 1 else -float(val_vals[0])

        model.save(model_path)

        return CandidateResult(
            candidate_id=candidate_id,
            architecture=arch,
            model_path=model_path,
            total_params=total_params,
            layer_names=layer_names,
            activations=layer_acts,
            final_loss=final_loss,
            final_metric=final_metric,
            final_accuracy=final_accuracy,
            final_mse=final_mse,
            val_performance=val_perf,
        )

    finally:
        if model is not None:
            del model
        tf.keras.backend.clear_session()
        gc.collect()



def run_random_search(
    data: PreparedData,
    output_dir: str,
    push_event: Callable,
    candidate_count: int,
    batch_size: int,
    epochs: int,
    start_candidate_id: int = 1,
):
    results: list[CandidateResult] = []
    cand_id = start_candidate_id

    for _ in range(candidate_count):
        arch = random_architecture()
        model_path = f"{output_dir}/candidate_{cand_id}.keras"
        result = _fit_candidate(data, arch, cand_id, model_path, push_event, batch_size, epochs)
        if result is not None:
            results.append(result)
            push_event(
                {
                    "candidate": result.candidate_id,
                    "final_accuracy": result.final_accuracy,
                    "final_mse": result.final_mse,
                    "final_loss": result.final_loss,
                    "total_params": result.total_params,
                },
                "result",
            )
        cand_id += 1

    return results



def run_evolutionary_search(
    data: PreparedData,
    output_dir: str,
    push_event: Callable,
    population_size: int,
    generations: int,
    batch_size: int,
    epochs: int,
    start_candidate_id: int = 1,
):
    population_size = max(4, min(6, population_size))
    generations = max(2, min(3, generations))

    population = [random_architecture() for _ in range(population_size)]
    results: list[CandidateResult] = []
    cand_id = start_candidate_id

    for gen in range(1, generations + 1):
        push_event({"message": f"Evolution generation {gen} started"}, "status")
        scored: list[tuple[float, dict]] = []

        for arch in population:
            model_path = f"{output_dir}/candidate_{cand_id}.keras"
            result = _fit_candidate(data, arch, cand_id, model_path, push_event, batch_size, epochs)
            if result is not None:
                results.append(result)
                scored.append((result.val_performance, arch))
                push_event(
                    {
                        "candidate": result.candidate_id,
                        "final_accuracy": result.final_accuracy,
                        "final_mse": result.final_mse,
                        "final_loss": result.final_loss,
                        "total_params": result.total_params,
                    },
                    "result",
                )
            cand_id += 1

        if not scored:
            break

        scored.sort(key=lambda x: x[0], reverse=True)
        survivors = [a for _, a in scored[: max(1, len(scored) // 2)]]

        new_population = survivors[:]
        while len(new_population) < population_size:
            new_population.append(mutate_architecture(random.choice(survivors)))
        population = new_population

    return results



def run_progressive_search(
    data: PreparedData,
    output_dir: str,
    push_event: Callable,
    candidate_limit: int,
    batch_size: int,
    epochs: int,
    start_candidate_id: int = 1,
):
    candidate_limit = max(1, candidate_limit)

    arch = {
        "dense_units": [random.choice(UNITS_CHOICES)],
        "activation": random.choice(ACTIVATIONS),
    }

    best_val = -np.inf
    results: list[CandidateResult] = []
    cand_id = start_candidate_id

    for _ in range(candidate_limit):
        model_path = f"{output_dir}/candidate_{cand_id}.keras"
        result = _fit_candidate(data, arch, cand_id, model_path, push_event, batch_size, epochs)
        if result is None:
            break

        results.append(result)
        push_event(
            {
                "candidate": result.candidate_id,
                "final_accuracy": result.final_accuracy,
                "final_mse": result.final_mse,
                "final_loss": result.final_loss,
                "total_params": result.total_params,
            },
            "result",
        )

        if result.val_performance > best_val:
            best_val = result.val_performance
            arch = grow_architecture(arch)
            push_event(
                {
                    "message": f"Progressive NAS improved at candidate {cand_id}; growing architecture",
                },
                "status",
            )
        else:
            push_event(
                {
                    "message": f"Progressive NAS stopped at candidate {cand_id}; no validation improvement",
                },
                "status",
            )
            break

        cand_id += 1

    return results
