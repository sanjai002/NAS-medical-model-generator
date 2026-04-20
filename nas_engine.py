"""
nas_engine.py  –  Neural Architecture Search Engine
=====================================================

This module implements the core NAS (Neural Architecture Search) logic.
It generates, trains, evaluates, and selects neural network architectures
automatically – the user only supplies the preprocessed data and some
high-level settings (how many candidates, which search strategy, etc.).

Three search strategies are provided:

  1. **Random Search** (`run_random_search`)
     Generates N completely independent random architectures, trains each
     one, and returns all results.  Simple but surprisingly effective.

  2. **Evolutionary Search** (`run_evolutionary_search`)
     Maintains a population of architectures.  Each generation: train
     them all → keep the top 50% → mutate survivors to fill the population
     back up → repeat for 2-3 generations.

  3. **Progressive Search** (`run_progressive_search`)
     Starts with a minimal 1-layer network and grows it deeper only when
     validation accuracy improves.  Stops early if adding depth does not
     help.

Architecture representation (the "search space"):
  An architecture is a plain Python dict with two keys:
    - "dense_units" : list[int]  – number of neurons in each hidden layer.
        Values are drawn from UNITS_CHOICES = [16, 32, 64, 128].
        Length ranges from 1 to 4 (1–4 hidden layers).
    - "activation"  : str        – activation function for hidden layers.
        Either "relu" or "tanh" (from ACTIVATIONS list).

  The output layer is always:
    - Dense(n_classes, activation="softmax")  for classification
    - Dense(1, activation="linear")           for regression

Real-time streaming:
  A custom Keras callback (StreamCallback) pushes per-batch and per-epoch
  training metrics into the SSE event queue so the browser can render
  live loss curves and weight-norm bar charts.

Key safety measures:
  - MAX_PARAMS = 1,000,000 : any architecture with more total parameters
    is skipped to avoid long training times.
  - Epochs are clamped to [1, 5] inside _fit_candidate.
  - EarlyStopping with patience=1 stops training if val_loss does not
    improve for one epoch, restoring the best weights.
  - tf.keras.backend.clear_session() and gc.collect() are called after
    each candidate to free GPU/CPU memory.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# Enables "X | Y" union syntax in type annotations for Python 3.9.
from __future__ import annotations

# gc (garbage collector) – explicitly called after each candidate to reclaim
#   memory from deleted Keras models and TensorFlow graph objects.
import gc
import time

# random – used to generate random architectures, pick random mutations,
#   and select random unit sizes / activation functions.
import random

# dataclass – defines the CandidateResult container.
from dataclasses import dataclass

# Callable type hint – used for the push_event callback signature.
from typing import Any, Callable

# NumPy – infinity value (np.inf) and norm calculation (np.linalg.norm).
import numpy as np

# TensorFlow / Keras – builds and trains the neural network models.
import tensorflow as tf

# Import the PreparedData dataclass from our data pipeline module.
# This is the input to every search function.
from data_pipeline import PreparedData


# ---------------------------------------------------------------------------
# Search-space constants
# ---------------------------------------------------------------------------

# The set of allowed neuron counts per hidden layer.
# Each hidden Dense layer will have one of these values as its unit count.
UNITS_CHOICES = [16, 32, 64, 128]

# The set of allowed activation functions for hidden layers.
# "relu" (Rectified Linear Unit) and "tanh" (Hyperbolic Tangent) are two
# standard non-linear activations used in feedforward networks.
ACTIVATIONS = ["relu", "tanh"]

# Maximum total parameter count allowed per model.  If a generated
# architecture exceeds this, it is silently skipped.  This prevents
# accidentally creating huge networks that would take too long to train.
MAX_PARAMS = 1_000_000


# ---------------------------------------------------------------------------
# CandidateResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class CandidateResult:
    """
    Stores the outcome of training a single candidate architecture.

    One instance is created per successfully trained model and appended to
    the results list returned by each search function.

    Attributes
    ----------
    candidate_id : int
        Sequential integer ID (1, 2, 3, …) identifying this candidate.
    architecture : dict
        The architecture dict that was used to build this model.
        E.g. {"dense_units": [64, 32], "activation": "relu"}.
    model_path : str
        File path where the trained .keras model was saved on disk.
    total_params : int
        Total number of trainable parameters in the model.
    layer_names : list[str]
        Names of the Dense layers (e.g. ["dense", "dense_1", "dense_2"]).
    activations : list[str]
        Activation function name for each Dense layer.
    final_loss : float
        Loss value on the held-out test set after training.
    final_metric : float
        Primary metric on the test set (accuracy for classification,
        MSE for regression).
    final_accuracy : float | None
        Test accuracy.  Set only for classification; None for regression.
    final_mse : float | None
        Test MSE.  Set only for regression; None for classification.
    val_performance : float
        A single number used to rank candidates.  For classification this
        is validation accuracy (higher = better).  For regression this is
        the negative of validation MSE (higher = better, since MSE should
        be minimized).  _choose_best() in app.py uses max() on this field.
    """
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
    val_metric: float
    val_performance: float
    training_time: float
    optimizer: str
    learning_rate: float


# ---------------------------------------------------------------------------
# Architecture generators
# ---------------------------------------------------------------------------

def random_architecture() -> dict:
    """
    Generate a completely random architecture within the search space.

    How it works:
      1. Pick a random depth: n_layers ∈ {1, 2, 3, 4}.
      2. For each layer, pick a random unit count from UNITS_CHOICES.
      3. Pick one activation function from ACTIVATIONS (shared by all layers).

    Returns
    -------
    dict
        {"dense_units": [list of ints], "activation": str}
        Example: {"dense_units": [128, 64], "activation": "relu"}
    """
    # random.randint(1, 4) returns an integer in [1, 4] inclusive.
    n_layers = random.randint(1, 4)

    return {
        # List comprehension creates n_layers random unit values.
        "dense_units": [random.choice(UNITS_CHOICES) for _ in range(n_layers)],
        # A single activation shared by every hidden layer in this candidate.
        "activation": random.choice(ACTIVATIONS),
    }



def mutate_architecture(arch: dict) -> dict:
    """
    Create a mutated copy of an existing architecture (used in evolutionary search).

    The function never modifies the input dict; it creates a new dict.

    Mutation types (one is chosen at random):
      - "add"   : Append a new random-sized layer (if depth < 4).
      - "remove": Remove a random layer (if depth > 1).
      - "change": Replace the unit count of one random layer.

    Additionally, with 30% probability the activation function is swapped
    (e.g. relu → tanh or tanh → relu).

    Parameters
    ----------
    arch : dict
        The parent architecture to mutate.
        {"dense_units": [...], "activation": "..."}

    Returns
    -------
    dict
        A new architecture dict with one structural mutation applied.
    """
    # Start with a shallow copy of the parent architecture.
    # list() creates a new list so we don't mutate the parent's list.
    out = {
        "dense_units": list(arch["dense_units"]),
        "activation": arch["activation"],
    }

    # Randomly pick which type of mutation to apply.
    mutation = random.choice(["add", "remove", "change"])

    if mutation == "add" and len(out["dense_units"]) < 4:
        # ADD: append a new layer with a random unit count.
        # Only allowed if the network has fewer than 4 layers.
        out["dense_units"].append(random.choice(UNITS_CHOICES))

    elif mutation == "remove" and len(out["dense_units"]) > 1:
        # REMOVE: delete one randomly chosen layer.
        # Only allowed if the network has more than 1 layer (must keep at least 1).
        # random.randrange returns an index in [0, len-1).
        out["dense_units"].pop(random.randrange(len(out["dense_units"])))

    else:
        # CHANGE: replace the unit count of one random layer.
        # This also handles fallback cases (e.g. "add" when already at 4 layers,
        # or "remove" when only 1 layer remains).
        idx = random.randrange(len(out["dense_units"]))
        out["dense_units"][idx] = random.choice(UNITS_CHOICES)

    # With 30% probability, also swap the activation function.
    # random.random() returns a float in [0.0, 1.0).
    if random.random() < 0.3:
        out["activation"] = random.choice(ACTIVATIONS)

    return out



def grow_architecture(arch: dict) -> dict:
    """
    Add one more hidden layer to an architecture (used in progressive search).

    If the architecture already has 4 layers (the maximum depth), it is
    returned unchanged.

    Parameters
    ----------
    arch : dict
        The current architecture to grow.

    Returns
    -------
    dict
        A new architecture with one additional random-sized layer appended.
        The activation is also re-randomized.
    """
    # Enforce maximum depth limit of 4 layers.
    if len(arch["dense_units"]) >= 4:
        return arch

    # Create a new dict with the existing layers plus one new random layer.
    out = {
        "dense_units": list(arch["dense_units"]) + [random.choice(UNITS_CHOICES)],
        "activation": random.choice(ACTIVATIONS),
    }
    return out


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_model(input_dim: int, output_dim: int, task: str, arch: dict) -> tf.keras.Model:
    """
    Construct a Keras Sequential-style model from an architecture dict.

    The model is built using the Functional API:
      Input(input_dim) → [Dense hidden layers] → Dense output layer

    Parameters
    ----------
    input_dim : int
        Number of input features (columns after preprocessing).
    output_dim : int
        Number of output neurons.  For classification = n_classes;
        for regression = 1.
    task : str
        "classification" or "regression".
    arch : dict
        Architecture specification with "dense_units" and "activation".

    Returns
    -------
    tf.keras.Model
        An uncompiled Keras model ready to be compiled and trained.
    """
    # Create the input tensor with shape (batch_size, input_dim).
    x_in = tf.keras.Input(shape=(input_dim,))
    x = x_in

    # Add hidden Dense layers according to the architecture spec.
    # Each layer uses the same activation function (arch["activation"]).
    for units in arch["dense_units"]:
        x = tf.keras.layers.Dense(units, activation=arch["activation"])(x)

    # Add the output layer depending on the task type.
    if task == "classification":
        # Softmax outputs a probability distribution over n classes.
        y = tf.keras.layers.Dense(output_dim, activation="softmax")(x)
    else:
        # Linear activation outputs a single continuous value for regression.
        y = tf.keras.layers.Dense(1, activation="linear")(x)

    # Wrap the input and output tensors into a Keras Model object.
    return tf.keras.Model(inputs=x_in, outputs=y)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_count_params(model: tf.keras.Model) -> int:
    """
    Safely count the total number of trainable parameters in a model.

    Uses model.count_params() which sums all weight matrix shapes.
    If it fails for any reason, returns MAX_PARAMS + 1 so the model
    will be treated as "too large" and skipped.

    Parameters
    ----------
    model : tf.keras.Model

    Returns
    -------
    int
        Total parameter count, or MAX_PARAMS + 1 on failure.
    """
    try:
        return int(model.count_params())
    except Exception:
        # Return above the limit so the candidate is skipped.
        return MAX_PARAMS + 1



def _metric_name(task: str) -> str:
    """
    Return the name of the primary tracked metric based on the task type.

    Parameters
    ----------
    task : str
        "classification" or "regression".

    Returns
    -------
    str
        "accuracy" for classification, "mse" for regression.
        These strings must match the metric names passed to model.compile().
    """
    return "accuracy" if task == "classification" else "mse"


# ---------------------------------------------------------------------------
# StreamCallback – real-time training event emitter
# ---------------------------------------------------------------------------

class StreamCallback(tf.keras.callbacks.Callback):
    """
    Custom Keras callback that pushes training progress to the SSE queue
    after every batch and every epoch.

    This is what makes the live dashboard work: the frontend's EventSource
    receives these events and updates the Chart.js loss curve and weight
    norm bar charts in real time.

    Attributes
    ----------
    push_event : Callable
        Reference to the app.py push_event() function.  Called as
        push_event(payload_dict, event_type_string).
    candidate_id : int
        ID of the candidate being trained (so the frontend knows which
        candidate's data is arriving).
    task : str
        "classification" or "regression" – determines which metric to read.
    batch_index : int
        Running counter of total batches processed (across all epochs).
        Incremented in on_train_batch_end.
    """

    def __init__(self, push_event: Callable, candidate_id: int, task: str):
        """
        Initialize the callback.

        Parameters
        ----------
        push_event : Callable
            The SSE event-pushing function from app.py.
        candidate_id : int
            Numeric ID of the candidate (1, 2, 3, …).
        task : str
            "classification" or "regression".
        """
        super().__init__()
        self.push_event = push_event
        self.candidate_id = candidate_id
        self.task = task
        self.batch_index = 0  # Cumulative batch counter (reset per candidate).

    def on_train_batch_end(self, batch, logs=None):
        """
        Called by Keras at the end of every training batch.

        What it does:
          1. Increments the batch counter.
          2. Iterates over all Dense layers in the model and computes
             the Frobenius norm of each layer's kernel (weight) matrix.
             This gives a single number summarizing the magnitude of
             weights in each layer – useful for monitoring training.
          3. Reads the batch loss and the primary metric (accuracy or MSE)
             from Keras' `logs` dict.
          4. Pushes a "step" event containing:
             - candidate   : which candidate is training
             - batch       : cumulative batch number
             - loss        : training loss this batch
             - batch_metric: accuracy or MSE this batch
             - weight_norms: list of {"layer": name, "norm": float}

        Parameters
        ----------
        batch : int
            Batch index within the current epoch (Keras provides this).
        logs : dict or None
            Dictionary of metric values for this batch, provided by Keras.
            Typically contains "loss" and possibly "accuracy" or "mse".
        """
        logs = logs or {}
        self.batch_index += 1

        # Compute weight norms for all Dense layers.
        # weight_norms is a list of dicts, one per Dense layer.
        weight_norms = []
        for lyr in self.model.layers:
            if isinstance(lyr, tf.keras.layers.Dense):
                # lyr.get_weights() returns [kernel_matrix, bias_vector].
                weights = lyr.get_weights()
                # Frobenius norm of the kernel matrix (weights[0]).
                # This is sqrt(sum of squared elements) – a scalar summary
                # of how "large" the weights are.
                kernel_norm = float(np.linalg.norm(weights[0])) if weights else 0.0
                weight_norms.append({"layer": lyr.name, "norm": kernel_norm})

        # Determine the metric key to read from logs.
        metric_key = _metric_name(self.task)  # "accuracy" or "mse"

        # Read the batch metric; fall back to loss if the metric is missing.
        batch_metric = float(logs.get(metric_key, logs.get("loss", 0.0)))

        # Push a "step" event to the SSE queue.
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
        """
        Called by Keras at the end of every epoch.

        Pushes an "epoch_end" event with validation loss and validation
        metric so the frontend can display per-epoch summaries.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index (Keras provides this).
        logs : dict or None
            Contains "val_loss", "val_accuracy" (or "val_mse"), etc.
        """
        logs = logs or {}
        metric_key = _metric_name(self.task)

        # Push an "epoch_end" event with validation metrics.
        self.push_event(
            {
                "candidate": self.candidate_id,
                "epoch": int(epoch + 1),                              # 1-based epoch number
                "val_loss": float(logs.get("val_loss", 0.0)),
                "val_metric": float(logs.get(f"val_{metric_key}", 0.0)),
            },
            "epoch_end",
        )


# ---------------------------------------------------------------------------
# Model compilation
# ---------------------------------------------------------------------------

def _compile_model(model: tf.keras.Model, task: str):
    """
    Compile a Keras model with the appropriate optimizer, loss, and metrics.

    Configuration per task type:
      - Classification:
          optimizer : Adam (adaptive learning rate)
          loss      : CategoricalCrossentropy (expects one-hot targets)
          metric    : accuracy
      - Regression:
          optimizer : Adam
          loss      : MeanSquaredError
          metric    : mse

    Parameters
    ----------
    model : tf.keras.Model
        The uncompiled model returned by build_model().
    task : str
        "classification" or "regression".
    """
    if task == "classification":
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),               # Adaptive learning rate
            loss=tf.keras.losses.CategoricalCrossentropy(),     # Multi-class log-loss
            metrics=["accuracy"],                                # Track accuracy during training
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.MeanSquaredError(),            # Squared error loss
            metrics=["mse"],                                     # Track MSE during training
        )


# ---------------------------------------------------------------------------
# Core candidate training function
# ---------------------------------------------------------------------------

def _fit_candidate(
    data: PreparedData,
    arch: dict,
    candidate_id: int,
    model_path: str,
    push_event: Callable,
    batch_size: int,
    epochs: int,
) -> CandidateResult | None:
    """
    Build, train, evaluate, and save ONE candidate neural network.

    This is the workhorse function called by all three search strategies.
    It handles the full lifecycle of a single candidate:
      1. Clear TensorFlow session (free memory from previous candidate).
      2. Build the model from the architecture dict.
      3. Count parameters; skip if over MAX_PARAMS.
      4. Compile the model.
      5. Push a "model_info" event so the frontend can show the architecture.
      6. Create tf.data Datasets for efficient batched training.
      7. Train with EarlyStopping (patience=1) and StreamCallback.
      8. Evaluate on test set and validation set.
      9. Save the trained model to disk as a .keras file.
     10. Return a CandidateResult with all metrics and metadata.
     11. In the finally block: delete the model, clear session, garbage collect.

    Parameters
    ----------
    data : PreparedData
        Preprocessed dataset arrays from data_pipeline.
    arch : dict
        Architecture dict: {"dense_units": [...], "activation": "..."}.
    candidate_id : int
        Sequential ID for this candidate (used in events and filenames).
    model_path : str
        Where to save the trained .keras file.
    push_event : Callable
        SSE event pusher from app.py.
    batch_size : int
        Number of samples per training batch.
    epochs : int
        Maximum number of training epochs (clamped to [1, 5]).

    Returns
    -------
    CandidateResult or None
        The result object if training was successful.
        None if the model was skipped (too many parameters).
    """
    # Clear any leftover TensorFlow graph/session from previous candidates.
    tf.keras.backend.clear_session()

    model = None
    try:
        # ── Step 1: Build the model from the architecture dict ──
        model = build_model(data.input_dim, data.output_dim, data.task, arch)

        # ── Step 2: Count total parameters and enforce the limit ──
        total_params = _safe_count_params(model)
        if total_params > MAX_PARAMS:
            # Send a "status" event to inform the frontend this candidate was skipped.
            push_event(
                {
                    "candidate": candidate_id,
                    "message": f"Skipped candidate {candidate_id}: parameters exceed limit",
                    "total_params": total_params,
                },
                "status",
            )
            return None  # Skip this candidate entirely.

        # ── Step 3: Compile the model (set optimizer, loss, metrics) ──
        _compile_model(model, data.task)

        # ── Step 4: Extract layer names and activation function names ──
        # These are sent to the frontend for the architecture visualization.
        # Only Dense layers are included (skip the Input layer).
        layer_names = [lyr.name for lyr in model.layers if isinstance(lyr, tf.keras.layers.Dense)]

        # For each Dense layer, get the activation function's name string.
        # getattr + hasattr chain handles layers that might not have .activation.
        layer_acts = [
            getattr(lyr, "activation", None).__name__ if hasattr(getattr(lyr, "activation", None), "__name__") else "linear"
            for lyr in model.layers
            if isinstance(lyr, tf.keras.layers.Dense)
        ]

        # ── Step 5: Push "model_info" event to the SSE stream ──
        # The frontend uses this to draw the architecture visualization boxes.
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

        # ── Step 6: Create tf.data.Dataset pipelines for training and validation ──
        # from_tensor_slices: wraps numpy arrays into a TF dataset.
        # .shuffle(1024): maintains a buffer of 1024 samples and draws randomly.
        # .batch(batch_size): groups samples into batches.
        # .prefetch(AUTOTUNE): overlaps data loading with training for speed.
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

        # ── Step 7: Set up callbacks ──
        # EarlyStopping: monitors val_loss.  If it doesn't improve for 1 epoch
        #   (patience=1), training stops and the best weights are restored.
        cb_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)

        # StreamCallback: our custom callback that sends live events to the browser.
        cb_stream = StreamCallback(push_event=push_event, candidate_id=candidate_id, task=data.task)

        # Measure wall-clock training time for UI/report comparison.
        fit_start = time.perf_counter()

        # ── Step 8: Train the model ──
        # epochs are clamped to [1, 5] as a safety measure.
        # verbose=0 suppresses Keras' own console output (we use SSE instead).
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=max(1, min(epochs, 5)),
            verbose=0,
            callbacks=[cb_early, cb_stream],
        )
        training_time = float(time.perf_counter() - fit_start)

        # ── Step 9: Evaluate on test set and validation set ──
        test_ds = tf.data.Dataset.from_tensor_slices((data.X_test, data.y_test)).batch(batch_size)

        # model.evaluate returns [loss, metric1, metric2, ...].
        eval_vals = model.evaluate(test_ds, verbose=0)   # Test set evaluation
        val_vals = model.evaluate(val_ds, verbose=0)      # Validation set evaluation

        # Extract loss and the primary metric from test evaluation.
        final_loss = float(eval_vals[0])       # Test loss
        final_metric = float(eval_vals[1]) if len(eval_vals) > 1 else float("nan")  # Test acc/mse

        # ── Step 10: Compute task-specific result values ──
        val_metric = float(val_vals[1]) if len(val_vals) > 1 else float(val_vals[0])

        if data.task == "classification":
            final_accuracy = final_metric       # Test accuracy (e.g. 0.85)
            final_mse = None                    # Not applicable for classification
            # val_perf is validation accuracy – used for ranking candidates.
            val_perf = val_metric
        else:
            final_accuracy = None               # Not applicable for regression
            final_mse = final_metric            # Test MSE value
            # For regression, val_perf is NEGATIVE MSE so that max() still picks
            # the best (lowest MSE) candidate.
            val_perf = -val_metric

        # ── Step 11: Save the trained model to disk ──
        model.save(model_path)

        # ── Step 12: Return the result ──
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
            val_metric=val_metric,
            val_performance=val_perf,
            training_time=training_time,
            optimizer="Adam",
            learning_rate=0.001,
        )

    finally:
        # ── Cleanup: always runs, even if training failed ──
        # Delete the model object to free its memory.
        if model is not None:
            del model
        # Clear the TF session (frees graph and variable memory).
        tf.keras.backend.clear_session()
        # Explicitly trigger Python garbage collection for any remaining objects.
        gc.collect()


# ---------------------------------------------------------------------------
# Search strategy 1: Random Search
# ---------------------------------------------------------------------------

def run_random_search(
    data: PreparedData,
    output_dir: str,
    push_event: Callable,
    candidate_count: int,
    batch_size: int,
    epochs: int,
    start_candidate_id: int = 1,
):
    """
    Train N independently generated random architectures and return all results.

    Algorithm:
      for i in 1..candidate_count:
        1. Generate a random architecture (random depth, random units, random activation).
        2. Train it using _fit_candidate().
        3. If training succeeded, append the result and push a "result" event.
        4. Move on to the next candidate.

    This is the simplest search strategy – no selection pressure or memory
    between candidates.  It works well when the search space is small.

    Parameters
    ----------
    data : PreparedData
        Preprocessed dataset from data_pipeline.
    output_dir : str
        Directory where .keras model files are saved.
    push_event : Callable
        SSE event pusher.
    candidate_count : int
        Number of random architectures to generate and train.
    batch_size : int
        Training batch size.
    epochs : int
        Maximum training epochs per candidate.
    start_candidate_id : int
        Starting ID number (default 1).

    Returns
    -------
    list[CandidateResult]
        Results for all candidates that trained successfully.
    """
    candidate_count = max(1, min(50, int(candidate_count)))

    results: list[CandidateResult] = []   # Accumulator for successful results.
    cand_id = start_candidate_id          # Current candidate ID counter.

    for _ in range(candidate_count):
        # Generate a brand new random architecture for each candidate.
        arch = random_architecture()

        # Construct the file path for saving – e.g. "uploads/candidate_1.keras".
        model_path = f"{output_dir}/candidate_{cand_id}.keras"

        # Train and evaluate this candidate.
        result = _fit_candidate(data, arch, cand_id, model_path, push_event, batch_size, epochs)

        if result is not None:
            # Training succeeded – collect the result and notify the frontend.
            results.append(result)
            push_event(
                {
                    "candidate": result.candidate_id,
                    "final_accuracy": result.final_accuracy,
                    "final_mse": result.final_mse,
                    "final_loss": result.final_loss,
                    "val_metric": result.val_metric,
                    "total_params": result.total_params,
                    "optimizer": result.optimizer,
                    "learning_rate": result.learning_rate,
                    "training_time": result.training_time,
                },
                "result",  # Event type that the frontend listens for.
            )
        # Increment ID even if the candidate was skipped (keeps IDs sequential).
        cand_id += 1

    return results


# ---------------------------------------------------------------------------
# Search strategy 2: Evolutionary Search
# ---------------------------------------------------------------------------

def run_evolutionary_search(
    data: PreparedData,
    output_dir: str,
    push_event: Callable,
    population_size: int,
    generations: int,
    batch_size: int,
    epochs: int,
    candidate_limit: int | None = None,
    start_candidate_id: int = 1,
):
    """
    Run an evolutionary (genetic) algorithm to search for the best architecture.

    Algorithm:
      1. Initialize a population of `population_size` random architectures.
      2. For each generation:
         a. Train every architecture in the population.
         b. Score them by validation performance.
         c. Keep the top 50% as survivors.
         d. Fill the population back to full size by mutating random
            survivors (add/remove/change a layer, possibly swap activation).
      3. Repeat for `generations` rounds.

    The idea is that good architectures "survive" and their mutations
    explore nearby variations – guided evolution toward better designs.

    Parameters
    ----------
    data : PreparedData
        Preprocessed dataset.
    output_dir : str
        Directory for saving .keras files.
    push_event : Callable
        SSE event pusher.
    population_size : int
        Number of architectures per generation (clamped to [4, 6]).
    generations : int
        Number of evolutionary generations (clamped to [2, 3]).
    batch_size : int
        Training batch size.
    epochs : int
        Max epochs per candidate.
    start_candidate_id : int
        Starting candidate ID.

    Returns
    -------
    list[CandidateResult]
        All successfully trained candidates across all generations.
    """
    # Respect the user-configured global candidate cap.
    max_candidates = max(1, min(50, int(candidate_limit if candidate_limit is not None else population_size)))
    population_size = max(1, min(int(population_size), max_candidates))
    generations = max(1, min(5, int(generations)))

    # Initialize the first generation with random architectures.
    population = [random_architecture() for _ in range(population_size)]

    results: list[CandidateResult] = []   # Collects ALL results across generations.
    cand_id = start_candidate_id

    for gen in range(1, generations + 1):
        if len(results) >= max_candidates:
            break

        remaining = max_candidates - len(results)
        if remaining <= 0:
            break

        # Notify the frontend that a new generation has started.
        push_event({"message": f"Evolution generation {gen} started"}, "status")

        # scored: list of (val_performance, architecture) tuples for this generation.
        scored: list[tuple[float, dict]] = []

        # Train every architecture in the current population.
        for arch in population[:remaining]:
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
                        "val_metric": result.val_metric,
                        "total_params": result.total_params,
                        "optimizer": result.optimizer,
                        "learning_rate": result.learning_rate,
                        "training_time": result.training_time,
                    },
                    "result",
                )
            cand_id += 1

        # If no candidates trained successfully, stop evolution.
        if not scored:
            break

        # ── Selection: keep the top 50% of architectures ──
        # Sort by val_performance descending (best first).
        scored.sort(key=lambda x: x[0], reverse=True)
        # Survivors = top half (at least 1).
        survivors = [a for _, a in scored[: max(1, len(scored) // 2)]]

        # ── Reproduction: fill population back up with mutated survivors ──
        new_population = survivors[:]   # Start with survivors as-is.
        while len(new_population) < population_size:
            # Pick a random survivor and create a mutated offspring.
            new_population.append(mutate_architecture(random.choice(survivors)))
        population = new_population

    return results


# ---------------------------------------------------------------------------
# Search strategy 3: Progressive Search
# ---------------------------------------------------------------------------

def run_progressive_search(
    data: PreparedData,
    output_dir: str,
    push_event: Callable,
    candidate_limit: int,
    batch_size: int,
    epochs: int,
    start_candidate_id: int = 1,
):
    """
    Incrementally grow a network, adding layers only when they improve performance.

    Algorithm:
      1. Start with a minimal architecture: 1 hidden layer, random units.
      2. Train it and record its validation performance.
      3. If performance improved over the previous best:
         a. Grow the architecture by adding one more layer (grow_architecture).
         b. Train the deeper version.
      4. If performance did NOT improve, stop immediately – adding more
         depth is not helping.
      5. Repeat up to `candidate_limit` times.

    This strategy is efficient because it stops as soon as depth stops
    helping, rather than training many unnecessary deep models.

    Parameters
    ----------
    data : PreparedData
        Preprocessed dataset.
    output_dir : str
        Directory for saving .keras files.
    push_event : Callable
        SSE event pusher.
    candidate_limit : int
        Maximum number of candidates to train before stopping.
    batch_size : int
        Training batch size.
    epochs : int
        Max epochs per candidate.
    start_candidate_id : int
        Starting candidate ID.

    Returns
    -------
    list[CandidateResult]
        All successfully trained candidates in this progressive run.
    """
    # Ensure at least 1 candidate is trained.
    candidate_limit = max(1, min(50, int(candidate_limit)))

    # Start with a minimal 1-layer architecture.
    arch = {
        "dense_units": [random.choice(UNITS_CHOICES)],  # Single layer, random units.
        "activation": random.choice(ACTIVATIONS),
    }

    # best_val tracks the best validation performance seen so far.
    # Initialized to -infinity so the first candidate always "improves".
    best_val = -np.inf

    results: list[CandidateResult] = []
    cand_id = start_candidate_id

    for _ in range(candidate_limit):
        model_path = f"{output_dir}/candidate_{cand_id}.keras"
        result = _fit_candidate(data, arch, cand_id, model_path, push_event, batch_size, epochs)

        # If training failed (e.g. too many params), stop the progressive search.
        if result is None:
            break

        results.append(result)
        push_event(
            {
                "candidate": result.candidate_id,
                "final_accuracy": result.final_accuracy,
                "final_mse": result.final_mse,
                "final_loss": result.final_loss,
                "val_metric": result.val_metric,
                "total_params": result.total_params,
                "optimizer": result.optimizer,
                "learning_rate": result.learning_rate,
                "training_time": result.training_time,
            },
            "result",
        )

        # ── Check if this candidate improved over the previous best ──
        if result.val_performance > best_val:
            # Improvement! Update the best score and grow the architecture.
            best_val = result.val_performance
            arch = grow_architecture(arch)  # Add one more layer for the next round.
            push_event(
                {
                    "message": f"Progressive NAS improved at candidate {cand_id}; growing architecture",
                },
                "status",
            )
        else:
            # No improvement – adding depth did not help. Stop searching.
            push_event(
                {
                    "message": f"Progressive NAS stopped at candidate {cand_id}; no validation improvement",
                },
                "status",
            )
            break

        cand_id += 1

    return results


def generate_readable_summary(
    best_model_info: dict[str, Any],
    all_models: list[dict[str, Any]],
    task_type: str,
) -> str:
    """
    Produce a human-readable best-model explanation for UI/report display.
    """
    if not all_models:
        return "No candidate models were available to summarize."

    fastest = min(all_models, key=lambda m: float(m.get("training_time", float("inf"))))
    smallest = min(all_models, key=lambda m: int(m.get("total_params", 10**18)))

    units = best_model_info.get("architecture", {}).get("dense_units", [])
    n_layers = len(units)
    units_txt = ", ".join(str(x) for x in units) if units else "unknown"

    activation = best_model_info.get("architecture", {}).get("activation")
    if not activation:
        acts = best_model_info.get("activations") or []
        activation = acts[0] if acts else "unknown"

    optimizer = best_model_info.get("optimizer", "Adam")
    lr = float(best_model_info.get("learning_rate", 0.001))
    params = int(best_model_info.get("total_params", 0))
    best_id = int(best_model_info.get("candidate", -1))
    train_sec = float(best_model_info.get("training_time", 0.0))

    if task_type == "classification":
        score = float(best_model_info.get("val_metric", best_model_info.get("final_accuracy", 0.0)))
        score_line = f"It achieved {score * 100:.1f}% validation accuracy in {train_sec:.1f} seconds."
        reason_line = "Compared to the other candidates, it delivered the strongest validation accuracy while keeping model complexity practical."
    else:
        score = float(best_model_info.get("val_metric", best_model_info.get("final_mse", 0.0)))
        rmse = score ** 0.5 if score >= 0 else float("nan")
        score_line = f"It achieved validation MAE/MSE quality with MSE {score:.4f} (RMSE {rmse:.4f}) in {train_sec:.1f} seconds."
        reason_line = "Compared to the other candidates, it reached the lowest validation error while maintaining reasonable complexity."

    badge_fragments: list[str] = []
    if best_id == int(fastest.get("candidate", -999999)):
        badge_fragments.append("Fastest Training")
    if best_id == int(smallest.get("candidate", -999999)):
        badge_fragments.append("Smallest Model")

    badges_txt = f" Badges: {', '.join(badge_fragments)}." if badge_fragments else ""

    return (
        f"Best Model: Model {best_id}. "
        f"This model contains {n_layers} hidden layers with {units_txt} neurons. "
        f"It uses the {activation} activation function and the {optimizer} optimizer. "
        f"The learning rate is {lr:g}. "
        f"The model has approximately {params:,} trainable parameters. "
        f"{score_line} "
        f"{reason_line}{badges_txt}"
    )
