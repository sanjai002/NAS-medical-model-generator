# AutoNAS – Neural Architecture Search Web Application

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Data Flow Diagrams](#data-flow-diagrams)
4. [Module Breakdown](#module-breakdown)
5. [Function Definitions & Detailed Documentation](#function-definitions--detailed-documentation)
6. [Key Concepts](#key-concepts)
7. [Request/Response Flows](#requestresponse-flows)
8. [Example Execution Flow](#example-execution-flow)

---

## Project Overview

**AutoNAS** is a web-based **Neural Architecture Search (NAS)** system that automatically discovers optimal neural network architectures for machine learning tasks. Users upload a dataset and choose a search strategy, then the system automatically:

1. **Preprocesses** the data (handling missing values, scaling, encoding)
2. **Generates** candidate neural network architectures
3. **Trains** and evaluates them in parallel
4. **Selects** the best performer
5. **Returns** a trained model ready for predictions

### Key Features

- **Automatic data preprocessing** – handles CSV/Excel files, numeric & categorical features
- **Three search strategies** – Random, Evolutionary, Progressive
- **Live training dashboard** – real-time metrics via Server-Sent Events (SSE)
- **Task detection** – automatically classifies problems as classification or regression
- **Model inference API** – predict on new data with a single HTTP request
- **Full pipeline persistence** – save/load models and preprocessing artifacts

### Technology Stack

- **Backend:** Flask (Python web framework)
- **ML/DL:** TensorFlow/Keras (neural network training)
- **Data Processing:** Pandas, NumPy, scikit-learn
- **Frontend:** HTML5, JavaScript, Chart.js (real-time charting)
- **Deployment:** Python 3.9+, pip dependencies

---

## System Architecture

### High-Level Component Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                         Flask Web Server (app.py)                   │
├──────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐              ┌──────────────────────────┐ │
│  │   Flask Routes      │              │   RuntimeState (shared)  │ │
│  │  (HTTP endpoints)   │◄─────────────┤  - trained_candidates   │ │
│  │                     │              │  - best_model_path      │ │
│  │  POST /upload       │              │  - feature_schema       │ │
│  │  GET  /stream       │              │  - task (class/regress) │ │
│  │  POST /predict      │              │  - run_in_progress      │ │
│  │  GET  /download_*   │              │  Protected by state_lock│ │
│  └────────┬────────────┘              └──────────────────────────┘ │
│           │                                                          │
│           └──────────────────────┬──────────────────────────────┐   │
│                                  ▼                              ▼   │
│                      ┌────────────────────┐    ┌─────────────────┐ │
│                      │ Background Thread  │    │  Model Cache    │ │
│                      │  (_train_worker)   │    │  (memory)       │ │
│                      └────────────────────┘    └─────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
                      │                  │
                      ▼                  ▼
        ┌──────────────────────┐  ┌──────────────────────┐
        │  data_pipeline.py    │  │  nas_engine.py       │
        │  (load & preprocess) │  │  (search & train)    │
        └──────────────────────┘  └──────────────────────┘
                      │                  │
                      ▼                  ▼
        ┌──────────────────────┐  ┌──────────────────────┐
        │  Raw Dataset         │  │  Keras Models        │
        │  (CSV/Excel)         │  │  (trained networks)  │
        └──────────────────────┘  └──────────────────────┘
```

### Threading Model

```
┌─────────────────────────────────────────────────────────────┐
│            Main Flask Thread (handles HTTP)                  │
│                                                               │
│  POST /upload                                                │
│    ├─ Save uploaded file                                     │
│    ├─ Validate NAS settings                                 │
│    ├─ Check if run in progress (set run_in_progress=True)  │
│    └─ SPAWN background thread → _train_worker              │
│       │                                                       │
│       └─────────────────────────────────────────┐            │
│                                                  ▼            │
│                          ┌─────────────────────────────────┐ │
│                          │ Background Thread (_train_worker)│ │
│                          │  (does all NAS work)           │ │
│                          │                                 │ │
│                          │ 1. Load & preprocess dataset   │ │
│                          │ 2. Run NAS search strategy     │ │
│                          │ 3. Train & evaluate candidates │ │
│                          │ 4. Select best model           │ │
│                          │ 5. Save model & report         │ │
│                          │ 6. Push SSE events to queue    │ │
│                          │ 7. Set run_in_progress=False   │ │
│                          └─────────────────────────────────┘ │
│                                  │                            │
│                                  ▼                            │
│       ┌──────────────────────────────────────────┐            │
│       │ Thread-Safe Queue (event_q)              │            │
│       │ - Stores JSON-serialized SSE events      │            │
│       │ - Consumed by GET /stream endpoint       │            │
│       └──────────────────────────────────────────┘            │
│                        │                                      │
│                        ▼                                      │
│       GET /stream                                            │
│       (SSE generator, reads from event_q)                    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Thread Safety Mechanisms

| Resource | Protection | Purpose |
|----------|-----------|---------|
| `RuntimeState` | `state_lock` | Protects access to trained_candidates, best_model_path, etc. |
| `event_q` (queue) | Built-in (Queue.Queue) | Thread-safe FIFO for SSE events |
| Cached Keras Model | `model_cache_lock` | Protects _cached_model and _cached_model_path |

---

## Data Flow Diagrams

### Complete Training Pipeline

```
User Uploads File (CSV/Excel)
           │
           ▼
┌─────────────────────────────────┐
│ POST /upload                     │
│ - Save file with timestamp      │
│ - Validate file format          │
│ - Parse NAS settings            │
│ - Spawn _train_worker thread    │
└─────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│ _train_worker (Background)      │
│                                  │
│ 1. Reset RuntimeState            │
│ 2. Cleanup old artifacts         │
└─────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│ load_and_prepare_dataset()       │
│ (data_pipeline.py)               │
│                                  │
│ INPUT:  filepath (CSV/Excel)    │
│                                  │
│ 1. Read file → Pandas DataFrame │
│ 2. Validate columns (≥2)        │
│ 3. Sample if too large (>20k)  │
│ 4. Drop rows with missing target │
│ 5. Separate X (features) & y    │
│ 6. Build feature schema          │
│ 7. Identify numeric vs cat cols  │
│ 8. Build preprocessing pipelines │
│ 9. Combine into ColumnTransformer│
│ 10. Detect task (class vs regr)  │
│ 11. Split: 60/20/20 train/val/te│
│ 12. Fit preprocessor on training │
│ 13. Transform all splits         │
│ 14. One-hot encode labels        │
│                                  │
│ OUTPUT: PreparedData object      │
│         - X_train, X_val, X_test │
│         - y_train, y_val, y_test │
│         - task, input_dim, output│
│         - preprocessor, encoder  │
│         - feature_schema         │
└─────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│ NAS Search Strategy              │
│ (nas_engine.py)                 │
│                                  │
│ ┌──────────────────────────────┐ │
│ │ RANDOM SEARCH               │ │
│ │ Generate N random archs     │ │
│ │ Train all independently     │ │
│ │ Return all results          │ │
│ └──────────────────────────────┘ │
│ OR                               │
│ ┌──────────────────────────────┐ │
│ │ EVOLUTIONARY SEARCH          │ │
│ │ 1. Init random population    │ │
│ │ 2. For each generation:      │ │
│ │    a. Train all archs        │ │
│ │    b. Keep top 50% survivors │ │
│ │    c. Mutate to fill pop     │ │
│ │ 3. Repeat 3 generations      │ │
│ └──────────────────────────────┘ │
│ OR                               │
│ ┌──────────────────────────────┐ │
│ │ PROGRESSIVE SEARCH            │ │
│ │ 1. Start: 1-layer network    │ │
│ │ 2. Train & evaluate          │ │
│ │ 3. If improved: grow arch    │ │
│ │ 4. Else: stop                │ │
│ │ 5. Repeat until no improve   │ │
│ └──────────────────────────────┘ │
└─────────────────────────────────┘
           │
           ├──────────────────────────────────────┐
           │                                      │
     For each candidate:                          │
           │                                      │
           ▼                                      ▼
┌──────────────────────────┐        ┌──────────────────────────┐
│ _fit_candidate()         │        │ SSE Events Pushed        │
│                          │        │                          │
│ 1. Clear TF session      │        │ - "status" events        │
│ 2. Build model           │        │ - "model_info" events    │
│ 3. Count params          │        │ - "step" events (batch)  │
│ 4. Skip if too large     │        │ - "epoch_end" events     │
│ 5. Compile model         │        │ - "result" events        │
│ 6. Create tf.data DS     │        │ - "done" event (final)   │
│ 7. Train with callbacks  │        │                          │
│    - EarlyStopping       │        │ → event_q queue          │
│    - StreamCallback      │        │ → GET /stream endpoint   │
│ 8. Evaluate on test set  │        │ → Browser receives live  │
│ 9. Save model to disk    │        │   updates               │
│ 10. Return CandidateResult│       └──────────────────────────┘
└──────────────────────────┘
           │
           └──────────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────────────┐
         │ _choose_best(results)          │
         │                                 │
         │ Returns candidate with highest  │
         │ val_performance                 │
         │ (accuracy for class,            │
         │  -MSE for regression)           │
         └────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────────────┐
         │ _build_report()                │
         │                                 │
         │ Create training_report.json:    │
         │ - timestamp                     │
         │ - task type                     │
         │ - best candidate summary        │
         │ - all candidates' metrics       │
         └────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────────────┐
         │ Save artifacts to disk:         │
         │ - best_pipeline.joblib         │
         │   (preprocessor + encoder)     │
         │ - training_report.json         │
         │ - best model already saved     │
         └────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────────────┐
         │ Update RuntimeState            │
         │ - trained_candidates dict      │
         │ - best_candidate_id            │
         │ - best_model_path              │
         │ - pipeline_path                │
         │ - run_in_progress = False      │
         └────────────────────────────────┘
```

### Prediction Pipeline

```
User navigates to /predict_page
           │
           ▼
┌─────────────────────────────┐
│ GET /predict_page           │
│ Serve predict.html          │
└─────────────────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Frontend calls:             │
│ GET /prediction_schema      │
└─────────────────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ /prediction_schema endpoint │
│                              │
│ Read RuntimeState:           │
│ - feature_schema             │
│ - task                       │
│ - best_candidate_id          │
│ - target_name                │
│                              │
│ Return JSON with fields      │
│ (name, type, min/max, etc)  │
└─────────────────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Frontend builds input form   │
│ dynamically based on schema  │
│                              │
│ - Number inputs for numeric  │
│ - Dropdowns for categorical  │
└─────────────────────────────┘
           │
    User fills form & submits
           │
           ▼
┌─────────────────────────────┐
│ POST /predict               │
│                              │
│ 1. Load best_pipeline.joblib│
│ 2. Load feature_schema       │
│ 3. Extract form values       │
│ 4. Build single-row DF       │
│ 5. Apply preprocessor        │
│ 6. Load best model (cached)  │
│ 7. Run model.predict()       │
│                              │
│ If classification:           │
│   - Get argmax of logits     │
│   - Decode to class label    │
│   - Build probability map    │
│                              │
│ If regression:               │
│   - Return raw prediction    │
│                              │
│ 8. Return JSON with result   │
└─────────────────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Frontend displays result     │
│                              │
│ Classification:              │
│   - Predicted class          │
│   - Probability for each     │
│                              │
│ Regression:                  │
│   - Predicted numeric value  │
└─────────────────────────────┘
```

---

## Module Breakdown

### 1. **app.py** – Flask Web Application

**Purpose:** Main web server; handles HTTP routes, coordinates all components, manages state.

**Key Responsibilities:**
- Serve HTML pages (/index.html, /predict.html)
- Upload handler – receive dataset & NAS config
- SSE streaming – push live training updates
- Prediction API – run inference on new data
- Download routes – retrieve models & reports
- Thread management – launch background training
- Model caching – avoid reloading on every prediction

**Global Objects:**
- `app` – Flask application instance
- `state` – RuntimeState dataclass (shared mutable state)
- `state_lock` – threading.Lock (protects state)
- `event_q` – queue.Queue (thread-safe SSE event queue)
- `_cached_model`, `_cached_model_path` – in-memory Keras model cache
- `model_cache_lock` – threading.Lock (protects model cache)

---

### 2. **data_pipeline.py** – Data Preprocessing

**Purpose:** Load raw datasets and transform them into ML-ready arrays.

**Key Responsibilities:**
- Read CSV/Excel files
- Auto-detect task (classification vs regression)
- Handle missing values
- Scale numeric features
- Encode categorical features
- Split into train/val/test
- Generate feature metadata schema

**Key Function:** `load_and_prepare_dataset(file_path, max_rows=20_000, random_state=42)`

**Output:** `PreparedData` dataclass with:
- X_train, X_val, X_test – feature arrays
- y_train, y_val, y_test – target arrays
- task – "classification" or "regression"
- preprocessor – fitted sklearn ColumnTransformer
- label_encoder – fitted LabelEncoder (class mapping)
- feature_schema – metadata for each column
- input_dim, output_dim – NN dimensions

---

### 3. **nas_engine.py** – Neural Architecture Search

**Purpose:** Generate, train, and evaluate candidate neural network architectures.

**Key Responsibilities:**
- Generate random/mutated architectures
- Build Keras models from architecture dicts
- Compile and train models
- Handle callbacks (EarlyStopping, StreamCallback)
- Evaluate on test set
- Run three search strategies:
  1. **Random Search** – N independent random architectures
  2. **Evolutionary Search** – genetic algorithm with mutations
  3. **Progressive Search** – incrementally deepen networks

**Architecture Representation:**
```python
{
    "dense_units": [128, 64, 32],      # List of layer sizes
    "activation": "relu"                 # Shared activation function
}
```

**Search Space:**
- Unit sizes: [16, 32, 64, 128]
- Activations: ["relu", "tanh"]
- Depth: 1–4 layers
- Parameter limit: 1,000,000

**Key Functions:**
- `random_architecture()` – generate random arch
- `mutate_architecture(arch)` – apply random mutation
- `grow_architecture(arch)` – add one layer
- `build_model(input_dim, output_dim, task, arch)` – Keras model
- `_fit_candidate(...)` – train one candidate
- `run_random_search(...)` – random search strategy
- `run_evolutionary_search(...)` – evolutionary strategy
- `run_progressive_search(...)` – progressive strategy

**StreamCallback Class:**
Custom Keras callback that pushes training metrics to SSE queue after every batch and epoch.

---

## Function Definitions & Detailed Documentation

### app.py Functions

#### `push_event(payload, event="update")`
**Purpose:** Serialize and queue an SSE event for browser delivery.

**Parameters:**
- `payload` (dict) – data to send
- `event` (str) – SSE event type name

**How it works:**
1. Wraps payload + event into JSON dict
2. Serializes to JSON string
3. Puts on thread-safe event_q queue
4. /stream endpoint reads from queue and yields to browser

**Example:**
```python
push_event({"message": "Training started"}, "status")
push_event(
    {"candidate": 1, "loss": 0.45, "accuracy": 0.92},
    "result"
)
```

---

#### `sse_stream()`
**Purpose:** Generator that yields SSE-formatted messages.

**How it works:**
1. Infinite loop reading from event_q
2. For each message, formats as SSE protocol:
   ```
   event: <type>
   data: <json>
   
   ```
3. Yields formatted lines to browser

**Connected to:** `/stream` route

---

#### `_train_worker(filepath, nas_type, candidates, batch_size, epochs)`
**Purpose:** Background thread function that executes full NAS pipeline.

**Steps:**
1. Reset RuntimeState
2. Cleanup old artifacts
3. Load & preprocess dataset via data_pipeline
4. Store feature metadata in RuntimeState
5. Run selected NAS strategy (random/evolutionary/progressive)
6. Select best candidate
7. Build training report JSON
8. Save preprocessing pipeline via joblib
9. Update RuntimeState with all results
10. Clear model cache
11. Push "done" SSE event

**Error Handling:**
- Catches exceptions, pushes "error" event
- Finally block always sets `run_in_progress = False`

---

#### `@app.route("/upload", methods=["POST"])`
**Purpose:** Receive dataset file and NAS settings; start background training.

**Parameters (multipart/form-data):**
- `file` – CSV or Excel file
- `nas_type` – "random", "evolutionary", or "progressive"
- `candidates` – number of candidates (1–6)
- `batch_size` – training batch size (16–32)
- `epochs` – max epochs per candidate (1–5)

**Validation:**
- Check file exists and is CSV/Excel
- Clamp numeric settings to valid ranges
- Reject if run already in progress (409 Conflict)

**Returns:**
- `{"status": "started", ...}` on success
- `{"error": "..."}` on failure (400/409)

---

#### `@app.route("/predict", methods=["POST"])`
**Purpose:** Run inference on user-provided data.

**Input (JSON or form-data):**
- Feature values with keys matching training features

**Processing:**
1. Load preprocessing pipeline from joblib
2. Extract feature values from request
3. Build single-row DataFrame
4. Transform using fitted ColumnTransformer
5. Load best model (cached)
6. Run model.predict()
7. Format result (class label or numeric)

**Returns:**
- Classification: `{"task": "classification", "prediction": "<label>", "probabilities": {...}}`
- Regression: `{"task": "regression", "prediction": 42.5}`
- Error: `{"error": "..."}` (400)

---

#### `_get_cached_model(model_path)`
**Purpose:** Load Keras model with in-memory caching.

**Thread-safe:** Uses `model_cache_lock`

**Behavior:**
- If model already cached and path matches: return cached copy
- If path changed: free old model, load new one
- Caches in module-level `_cached_model` variable

**Benefit:** Avoids I/O overhead on repeated /predict calls

---

### data_pipeline.py Functions

#### `load_and_prepare_dataset(file_path, max_rows=20_000, random_state=42) → PreparedData`
**Purpose:** Single entry point for full data preprocessing pipeline.

**Steps:**

1. **Read file:**
   - Detect CSV vs Excel from extension
   - Load into Pandas DataFrame

2. **Validate:**
   - Minimum 2 columns (features + target)
   - At least 1 row

3. **Sample:**
   - If > max_rows: random sample of max_rows

4. **Clean target:**
   - Drop rows with missing target values
   - (Features are imputed later)

5. **Split features/target:**
   - Features (X_df) = all columns except last
   - Target (y) = last column

6. **Build feature schema:**
   - Records metadata for each feature
   - Type (numeric/categorical)
   - Min/max/mean for numeric
   - Unique values for categorical

7. **Identify column types:**
   - numeric_cols = columns with number/bool dtype
   - categorical_cols = rest

8. **Build preprocessing pipelines:**
   - **Numeric:** Impute(median) → StandardScaler
   - **Categorical:** Impute(mode) → OneHotEncoder

9. **Combine into ColumnTransformer**

10. **Detect task type:**
    - Non-numeric dtype → classification
    - Numeric with ≤20 unique → classification
    - Many unique numeric → regression

11. **Classification path:**
    - Encode labels to integers with LabelEncoder
    - Stratified split 80/20 train+val / test
    - Split train+val 75/25 train / val (= 60/20/20 overall)
    - Fit preprocessor on train
    - Transform all three splits
    - One-hot encode integer labels for Keras

12. **Regression path:**
    - Coerce target to float
    - Random split 80/20 train+val / test
    - Random split 75/25 → 60/20/20
    - Fit preprocessor on train
    - Transform all three splits
    - Reshape targets to (n, 1)

13. **Return:**
    - PreparedData with all arrays + metadata

---

#### `_detect_task(y) → str`
**Purpose:** Automatically determine classification vs regression.

**Logic:**
- If target dtype is non-numeric (strings) → "classification"
- If ≤20 unique numeric values → "classification"
- Otherwise → "regression"

---

#### `_build_feature_schema(df_features) → list[dict]`
**Purpose:** Create metadata describing each feature.

**For numeric columns:**
```python
{
    "name": "age",
    "type": "numeric",
    "min": 18.0,
    "max": 85.0,
    "mean": 45.3,
    "example": 45.3
}
```

**For categorical columns:**
```python
{
    "name": "color",
    "type": "categorical",
    "choices": ["blue", "red", "green"],
    "example": "blue"
}
```

**Used by:** Frontend to dynamically build prediction form

---

#### `_build_one_hot_encoder() → OneHotEncoder`
**Purpose:** Create sklearn OneHotEncoder compatible across versions.

**Compatibility:** Handles parameter name change from `sparse` (old) to `sparse_output` (new)

**Config:**
- `sparse_output=False` – output dense array
- `handle_unknown="ignore"` – allow unseen categories

---

#### `_safe_classification_split(...) → tuple`
**Purpose:** Stratified split with fallback for edge cases.

**Behavior:**
1. Try stratified split (preserves class ratios)
2. If fails (rare class): fall back to random split

---

### nas_engine.py Functions

#### `random_architecture() → dict`
**Purpose:** Generate single random architecture.

**Process:**
1. Pick random depth: 1–4 layers
2. For each layer: pick random unit count
3. Pick random activation (shared by all)

**Example output:**
```python
{"dense_units": [64, 128, 32], "activation": "relu"}
```

---

#### `mutate_architecture(arch) → dict`
**Purpose:** Create mutated copy for evolutionary search.

**Mutation types (one chosen randomly):**
- **"add"** – append new random layer (if depth < 4)
- **"remove"** – delete random layer (if depth > 1)
- **"change"** – replace layer's unit count

**Additionally:**
- 30% chance to swap activation function

**Never modifies input** – returns new dict

---

#### `grow_architecture(arch) → dict`
**Purpose:** Add one layer for progressive search.

**Behavior:**
- If already 4 layers: return unchanged
- Else: append random-sized layer + re-randomize activation

---

#### `build_model(input_dim, output_dim, task, arch) → tf.keras.Model`
**Purpose:** Construct uncompiled Keras model from architecture dict.

**Architecture:**
```
Input(input_dim) 
  → Dense(units[0], activation)
  → Dense(units[1], activation)
  → ...
  → Dense(output_dim, task_activation)
  → Output
```

**Task-specific output:**
- Classification: Dense(output_dim, "softmax")
- Regression: Dense(1, "linear")

**Returns:** Uncompiled tf.keras.Model

---

#### `_compile_model(model, task)`
**Purpose:** Configure optimizer, loss, metrics for a model.

**Classification:**
- Optimizer: Adam
- Loss: CategoricalCrossentropy
- Metric: accuracy

**Regression:**
- Optimizer: Adam
- Loss: MeanSquaredError
- Metric: mse

---

#### `_fit_candidate(data, arch, candidate_id, model_path, push_event, batch_size, epochs) → CandidateResult | None`
**Purpose:** Train one candidate architecture from start to finish.

**Steps:**

1. **Clear session:** Free memory from previous candidate
2. **Build model:** Via build_model()
3. **Count params:** Skip if > 1M
4. **Compile:** Via _compile_model()
5. **Extract layer info:** Names & activations
6. **Push model_info event**
7. **Create tf.data Datasets:**
   - Train: shuffle + batch + prefetch
   - Validation: batch + prefetch
8. **Set up callbacks:**
   - EarlyStopping (patience=1)
   - StreamCallback (push events per batch/epoch)
9. **Train:** model.fit() with epochs clamped to [1, 5]
10. **Evaluate:** On test set (→ final_loss, final_metric)
11. **Compute task-specific fields:**
    - Classification: final_accuracy, val_performance = val_acc
    - Regression: final_mse, val_performance = -val_mse
12. **Save model:** To disk as .keras file
13. **Return:** CandidateResult dataclass
14. **Cleanup:** Delete model, clear session, gc.collect()

**Returns:**
- CandidateResult on success
- None if model skipped (too many params)

---

#### `class StreamCallback(tf.keras.callbacks.Callback)`
**Purpose:** Custom callback pushing training metrics to SSE queue.

**Methods:**

**`on_train_batch_end(batch, logs)`**
- Called after every batch
- Pushes "step" event with:
  - candidate ID
  - batch index
  - loss
  - batch metric (accuracy/MSE)
  - weight norms per Dense layer

**`on_epoch_end(epoch, logs)`**
- Called after every epoch
- Pushes "epoch_end" event with:
  - epoch number
  - validation loss
  - validation metric

---

#### `run_random_search(data, output_dir, push_event, candidate_count, batch_size, epochs) → list[CandidateResult]`
**Purpose:** Train N independently random architectures.

**Algorithm:**
```
for i in 1..candidate_count:
    arch = random_architecture()
    result = _fit_candidate(...)
    if result:
        results.append(result)
        push_event("result")
```

**Returns:** All CandidateResult objects from successful training

---

#### `run_evolutionary_search(data, output_dir, push_event, population_size, generations, batch_size, epochs) → list[CandidateResult]`
**Purpose:** Genetic algorithm search with selection & mutation.

**Algorithm:**
```
population = [random_architecture() for _ in range(population_size)]

for generation in 1..generations:
    for arch in population:
        result = _fit_candidate(arch)  # Train
        results.append(result)
    
    # Selection: keep top 50%
    survivors = sorted(results)[:len(results)//2]
    
    # Reproduction: fill population with mutated survivors
    new_population = survivors + [mutate(random_survivor()) 
                                   for _ in range(shortfall)]
    population = new_population
```

**Clamping:**
- population_size: [4, 6]
- generations: [2, 3]

**Returns:** All CandidateResult objects across all generations

---

#### `run_progressive_search(data, output_dir, push_event, candidate_limit, batch_size, epochs) → list[CandidateResult]`
**Purpose:** Incrementally deepen networks while improving.

**Algorithm:**
```
arch = minimal_1_layer_architecture()
best_val = -infinity

for i in 1..candidate_limit:
    result = _fit_candidate(arch)
    results.append(result)
    
    if result.val_performance > best_val:
        # Improved! Grow and continue
        best_val = result.val_performance
        arch = grow_architecture(arch)
    else:
        # No improvement, stop
        break
```

**Stops early** if depth stops helping

**Returns:** CandidateResult objects until improvement stops

---

## Key Concepts

### PreparedData Dataclass
Container holding all preprocessed data:
- **Arrays:** X_train, X_val, X_test, y_train, y_val, y_test
- **Metadata:** task, input_dim, output_dim, target_name
- **Artifacts:** preprocessor (ColumnTransformer), label_encoder (LabelEncoder)
- **Schema:** feature_schema (metadata per column), feature_columns (names)

### CandidateResult Dataclass
Holds outcome of training one candidate:
- **Architecture info:** architecture dict, layer_names, activations, total_params
- **Performance:** final_loss, final_metric, final_accuracy, final_mse
- **Ranking:** val_performance (higher = better)
- **File:** model_path (where .keras file saved)

### RuntimeState Dataclass
Global shared state protected by `state_lock`:
- **Results:** trained_candidates dict, best_candidate_id, best_model_path
- **Files:** report_path, pipeline_path
- **Metadata:** feature_schema, feature_columns, task, target_name
- **Status:** run_in_progress (bool)

### Architecture Search Space
- **Representation:** `{"dense_units": [...], "activation": "..."}`
- **Unit choices:** [16, 32, 64, 128]
- **Activation choices:** ["relu", "tanh"]
- **Depth:** 1–4 hidden layers
- **Parameter limit:** 1,000,000 (skips oversized models)

### Two Preprocessing Pipelines

**For Features:**
```
numeric_pipeline:
  - SimpleImputer(strategy="median")
  - StandardScaler()

categorical_pipeline:
  - SimpleImputer(strategy="most_frequent")
  - OneHotEncoder()

Combined via ColumnTransformer
```

**For Target (Classification):**
```
LabelEncoder: {"No": 0, "Yes": 1}
Then to_categorical() for Keras: [1, 0] or [0, 1]
```

---

## Request/Response Flows

### POST /upload

**Request:**
```
multipart/form-data:
  file: <uploaded CSV/Excel>
  nas_type: "random" | "evolutionary" | "progressive"
  candidates: "4"
  batch_size: "32"
  epochs: "3"
```

**Response (200 Success):**
```json
{
  "status": "started",
  "nas_type": "random",
  "candidates": 4,
  "batch_size": 32,
  "epochs": 3
}
```

**Response (400 Bad Request):**
```json
{"error": "No file uploaded"}
{"error": "Only CSV and Excel files are supported"}
{"error": "Invalid NAS type"}
```

**Response (409 Conflict):**
```json
{"error": "A training run is already in progress"}
```

**Side Effects:**
- File saved to uploads/ with timestamp prefix
- Background thread spawned to run NAS pipeline
- SSE events pushed to /stream endpoint

---

### GET /stream

**Protocol:** Server-Sent Events (text/event-stream)

**Message format:**
```
event: <type>
data: <json>

```

**Event types:**
- `status` – progress messages
- `model_info` – candidate architecture details
- `step` – per-batch training metrics
- `epoch_end` – per-epoch validation metrics
- `result` – final candidate results
- `done` – training completed
- `error` – error message

**Example stream:**
```
event: status
data: {"message": "Starting random NAS..."}

event: status
data: {"message": "Dataset ready. task=classification, samples=120"}

event: model_info
data: {"candidate": 1, "arch": {...}, "total_params": 45000}

event: step
data: {"candidate": 1, "batch": 10, "loss": 0.45, "batch_metric": 0.88, "weight_norms": [...]}

event: epoch_end
data: {"candidate": 1, "epoch": 1, "val_loss": 0.42, "val_metric": 0.89}

event: result
data: {"candidate": 1, "final_accuracy": 0.91, "final_loss": 0.35}

event: done
data: {"message": "NAS completed", "best_candidate": 1, "best_model": {...}}
```

---

### GET /prediction_schema

**Response (200, ready):**
```json
{
  "ready": true,
  "task": "classification",
  "target_name": "disease",
  "best_candidate_id": 3,
  "fields": [
    {
      "name": "age",
      "type": "numeric",
      "min": 18.0,
      "max": 85.0,
      "mean": 45.3,
      "example": 45.3
    },
    {
      "name": "gender",
      "type": "categorical",
      "choices": ["F", "M"],
      "example": "M"
    }
  ]
}
```

**Response (200, not ready):**
```json
{
  "ready": false,
  "message": "Model is not ready yet"
}
```

---

### POST /predict

**Request (JSON):**
```json
{
  "age": 45,
  "gender": "M",
  "cholesterol": 200
}
```

**Request (form-data):**
```
age: 45
gender: M
cholesterol: 200
```

**Response (Classification, 200):**
```json
{
  "task": "classification",
  "prediction": "No",
  "probabilities": {
    "No": 0.85,
    "Yes": 0.15
  }
}
```

**Response (Regression, 200):**
```json
{
  "task": "regression",
  "prediction": 42.7
}
```

**Response (Error, 400):**
```json
{"error": "No trained model available"}
```

---

### GET /download_best

**Response:** Binary file (best model's .keras file) with header:
```
Content-Disposition: attachment; filename="candidate_X.keras"
```

---

### GET /download_report

**Response:** JSON file (training_report.json)
```json
{
  "timestamp": "2026-04-09T12:34:56Z",
  "task": "classification",
  "best_candidate": 3,
  "best_summary": {
    "architecture": {"dense_units": [64, 32], "activation": "relu"},
    "total_params": 4512,
    "final_accuracy": 0.92,
    "final_loss": 0.28
  },
  "candidates": [
    {
      "candidate": 1,
      "architecture": {...},
      "final_accuracy": 0.87,
      ...
    },
    ...
  ]
}
```

---

## Example Execution Flow

### Scenario: User trains classification model on Heart Disease dataset

#### Step 1: User Navigates to Frontend
```
Browser requests: GET /
Flask returns: index.html (training page)
```

#### Step 2: User Uploads Dataset
```
User selects file: heart_disease.csv (120 rows, 13 features + 1 target)
User selects: nas_type="evolutionary", candidates=4, epochs=3
User clicks: Upload

Browser sends: POST /upload
  - Multipart file upload
  - Form fields with NAS settings
```

#### Step 3: Flask Validates and Spawns Background Thread
```
Flask validates:
  ✓ File is CSV
  ✓ NAS type is valid
  ✓ No run in progress

Flask saves: uploads/20260409_153000_heart_disease.csv

Flask spawns: _train_worker("uploads/...", "evolutionary", 4, 32, 3)
  (runs in daemon thread)

Flask returns: HTTP 200 with {"status": "started", ...}

Browser starts listening: EventSource('/stream')
```

#### Step 4: Background Thread Executes NAS Pipeline
```
_train_worker thread:

1. Reset RuntimeState
   - Clear trained_candidates dict
   - Set run_in_progress = True

2. Load & preprocess dataset
   - Read CSV: 120 rows × 13 features + 1 target
   - Task detection: target is 0/1 integers → classification
   - Split: 72 train, 24 val, 24 test
   - Preprocessing:
     - Numeric (age, cholesterol, ...):
       → Impute missing with median
       → StandardScaler
     - Categorical (gender, ...):
       → Impute with mode
       → OneHotEncoder
   - Output_dim = 2 (binary classification)

3. Push "status" event
   Event: "Dataset ready. task=classification, samples=120"
   Browser sees live update

4. Evolutionary Search (4 generations × 4 population)
   
   Generation 1:
   ┌─ Architecture 1: {dense_units: [64, 32], activation: "relu"}
   │  - Build model: Input → Dense(64,relu) → Dense(32,relu) → Dense(2,softmax)
   │  - Total params: ~4500
   │  - Train for ≤3 epochs with EarlyStopping
   │  - StreamCallback pushes:
   │    - per-batch: loss, accuracy, weight_norms
   │    - per-epoch: val_loss, val_accuracy
   │  - After training: accuracy=0.87, loss=0.38
   │  - Save: uploads/candidate_1.keras
   │  - Push "result" event → Browser updates chart
   │
   ├─ Architecture 2: {dense_units: [128], activation: "tanh"}
   │  - Training... accuracy=0.79
   │  - Save: uploads/candidate_2.keras
   │
   ├─ Architecture 3: {dense_units: [32, 64, 32], activation: "relu"}
   │  - Training... accuracy=0.90 ← Best so far
   │
   └─ Architecture 4: {dense_units: [16, 32], activation: "relu"}
      - Training... accuracy=0.85
   
   Selection: Keep top 50% → [Arch3, Arch1]
   
   Reproduction: Mutate 2 survivors to fill population
   - Mutated_Arch3: Remove a layer, swap activation
   - Mutated_Arch1: Add a layer
   
   Generation 2:
   ┌─ Arch3 (retrained): accuracy=0.91
   ├─ Arch1 (retrained): accuracy=0.87
   ├─ Mutated_Arch3: accuracy=0.88
   └─ Mutated_Arch1: accuracy=0.92 ← New best!
   
   (Continue generation 3, ...)

5. After evolution complete, results list has ~12 trained candidates

6. _choose_best()
   - Find candidate with highest val_performance
   - Best = Candidate #10 with 0.92 validation accuracy

7. _build_report()
   - Create training_report.json with all metrics

8. Save artifacts:
   - best_pipeline.joblib: preprocessor + label_encoder
   - training_report.json: full results

9. Update RuntimeState
   - trained_candidates = {1: {...}, 2: {...}, ..., 12: {...}}
   - best_candidate_id = 10
   - best_model_path = "uploads/candidate_10.keras"
   - run_in_progress = False

10. Push "done" event
    Event: {"message": "NAS completed", "best_candidate": 10, ...}
    Browser sees completion message
```

#### Step 5: User Views Results
```
Frontend receives all SSE events in real-time
- Charts update as each candidate trains
- Final summary shows best model

User can:
- Download best_pipeline.joblib
- Download training_report.json
- Download candidate_10.keras (best model)
- Click: "Go to Prediction Page"
```

#### Step 6: User Makes Predictions
```
Browser navigates: GET /predict_page
Flask returns: predict.html

Frontend calls: GET /prediction_schema
Response:
{
  "ready": true,
  "task": "classification",
  "fields": [
    {"name": "age", "type": "numeric", ...},
    {"name": "gender", "type": "categorical", ...},
    ...
  ]
}

Frontend dynamically builds HTML form:
- Number input for age
- Dropdown for gender
- etc.

User fills form:
  age: 50
  gender: M
  cholesterol: 250
  ...

User clicks: Predict

Browser sends: POST /predict (JSON body)

Flask processes:
1. Load best_pipeline.joblib
2. Extract form values
3. Build DataFrame with correct column order
4. Apply ColumnTransformer
   - age: (50 - mean) / std
   - gender: OneHotEncode("M") → [0, 1] or [1, 0]
   - cholesterol: (250 - mean) / std
5. Load best model (from cache or disk)
6. model.predict() → [[0.15, 0.85]]
7. Argmax → class_idx = 1
8. label_encoder.inverse_transform([1]) → "Yes"
9. Return JSON:
   {
     "task": "classification",
     "prediction": "Yes",
     "probabilities": {
       "No": 0.15,
       "Yes": 0.85
     }
   }

Frontend displays result:
  Prediction: Yes (85% confidence)
  
User is done!
```

---

## Architecture Visualization

### Neural Network Output by NAS

For a candidate with architecture:
```python
{"dense_units": [64, 32], "activation": "relu"}
```

The resulting Keras model (for classification with 13 input features):

```
Input (13 features)
  │
  ├─ StandardScaler / OneHotEncoder
  │
  ▼
┌─────────────────┐
│  Dense Layer 1  │
│  Units: 64      │
│  Activation:    │
│      ReLU       │
│  Params: 832    │  (13×64 + 64 bias)
└─────────────────┘
  │
  ▼
┌─────────────────┐
│  Dense Layer 2  │
│  Units: 32      │
│  Activation:    │
│      ReLU       │
│  Params: 2080   │  (64×32 + 32 bias)
└─────────────────┘
  │
  ▼
┌─────────────────┐
│  Output Layer   │
│  Units: 2       │
│  Activation:    │
│   Softmax       │
│  Params: 66     │  (32×2 + 2 bias)
└─────────────────┘
  │
  ▼
Output (2 probabilities: [P_No, P_Yes])

Total Parameters: ~2,978
```

---

## Summary of Data Types

| Type | Example | Used In |
|------|---------|---------|
| PreparedData | dataclass with X_train, X_val, X_test, y_train, y_val, y_test, ... | Data loading output |
| CandidateResult | dataclass with architecture, final_accuracy, model_path, ... | Training result per candidate |
| RuntimeState | dataclass with trained_candidates, best_candidate_id, ... | Shared mutable state |
| architecture | `{"dense_units": [64, 32], "activation": "relu"}` | NAS representation |
| ColumnTransformer | fitted sklearn preprocessor | Feature preprocessing |
| LabelEncoder | maps "No"↔0, "Yes"↔1 | Classification label mapping |
| tf.keras.Model | compiled+trained neural network | Inference |
| Keras Callback | StreamCallback | Real-time training event streaming |

---

This documentation provides a complete reference for understanding, extending, and debugging the AutoNAS system. All major functions, data flows, and architectural concepts are covered with examples and diagrams.
