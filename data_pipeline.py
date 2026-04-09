"""
data_pipeline.py  –  Dataset Loading & Preprocessing Module
============================================================

This module is the first stage of the AutoNAS system.  It takes a raw
dataset file (CSV or Excel) uploaded by the user, and transforms it into
clean, numeric arrays that can be fed directly into TensorFlow/Keras
neural network models.

The module handles the full pipeline:
  1. Reading the file from disk (CSV or Excel).
  2. Sampling down to a maximum row limit to keep training fast.
  3. Automatically detecting whether the problem is classification or
     regression based on the target column.
  4. Building separate preprocessing pipelines for numeric columns
     (median imputation → StandardScaler) and categorical columns
     (mode imputation → OneHotEncoding).
  5. Splitting data into train / validation / test sets (60 / 20 / 20).
  6. One-hot encoding the target labels for classification tasks.
  7. Building a "feature schema" (metadata about each column) that the
     prediction UI uses later to generate a dynamic input form.
  8. Returning everything packaged inside a PreparedData dataclass.

Key concepts:
  - PreparedData : a dataclass that bundles all arrays, metadata, and the
    fitted sklearn preprocessor so downstream code can use a single object.
  - ColumnTransformer : the sklearn object that routes numeric columns
    through one pipeline and categorical columns through another.
  - Feature schema : a list of dicts describing each feature (name, type,
    min/max/mean for numeric, choices for categorical).  This is serialized
    to JSON and used by the predict.html page.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# Enables "X | Y" union type syntax in annotations for Python 3.9.
from __future__ import annotations

# dataclass decorator – used to define PreparedData with typed fields.
from dataclasses import dataclass

# Any type hint – used for generic dict values.
from typing import Any

# NumPy – numerical array operations; all feature/target data is stored as
# np.ndarray arrays so TensorFlow can consume them directly.
import numpy as np

# Pandas – reads CSV/Excel files into DataFrames and provides column-level
# dtype inspection, missing-value handling, and sampling utilities.
import pandas as pd

# TensorFlow – used here only for tf.keras.utils.to_categorical(), which
# converts integer class labels into one-hot encoded vectors.
import tensorflow as tf

# scikit-learn components for the preprocessing pipeline:
#   ColumnTransformer  – routes different column groups to different pipelines.
#   SimpleImputer      – fills missing values (median for numeric, mode for categorical).
#   train_test_split   – splits arrays into train/test subsets.
#   Pipeline           – chains sequential transforms (imputer → scaler/encoder).
#   LabelEncoder       – maps string class labels to integer indices (0, 1, 2, …).
#   OneHotEncoder      – converts categorical strings into binary dummy columns.
#   StandardScaler     – zero-mean, unit-variance normalization for numeric features.
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum number of rows the system will use.  If the uploaded dataset has
# more rows than this, a random sample of MAX_ROWS_DEFAULT rows is taken.
# This keeps training time reasonable for a web-based demo.
MAX_ROWS_DEFAULT = 20_000


# ---------------------------------------------------------------------------
# PreparedData dataclass
# ---------------------------------------------------------------------------

@dataclass
class PreparedData:
    """
    Container that holds everything produced by load_and_prepare_dataset().

    Attributes
    ----------
    X_train : np.ndarray
        2-D float32 array of preprocessed training features.
        Shape: (n_train_samples, input_dim).
    X_val : np.ndarray
        2-D float32 array of preprocessed validation features.
        Shape: (n_val_samples, input_dim).
    X_test : np.ndarray
        2-D float32 array of preprocessed test features.
        Shape: (n_test_samples, input_dim).
    y_train : np.ndarray
        Training labels.  For classification this is a one-hot encoded
        float32 array of shape (n_train, n_classes).  For regression it
        is shape (n_train, 1).
    y_val : np.ndarray
        Validation labels (same encoding as y_train).
    y_test : np.ndarray
        Test labels (same encoding as y_train).
    task : str
        Either "classification" or "regression".
    input_dim : int
        Number of features after preprocessing (i.e. X_train.shape[1]).
        This is passed to the NAS engine to set the Input layer size.
    output_dim : int
        Number of output neurons.  For classification = number of classes;
        for regression = 1.
    preprocessor : ColumnTransformer
        The fitted sklearn ColumnTransformer.  It is saved to disk via
        joblib so the /predict route can reuse the exact same transform.
    label_encoder : LabelEncoder | None
        Fitted LabelEncoder that maps class labels ↔ integer indices.
        None for regression tasks.
    feature_columns : list[str]
        Original column names of the feature DataFrame (before encoding).
    feature_schema : list[dict[str, Any]]
        Metadata about each feature column.  Each dict contains:
          - "name"    : column name (str)
          - "type"    : "numeric" or "categorical"
          - "min", "max", "mean", "example"  (for numeric)
          - "choices", "example"              (for categorical)
        This schema is sent to the frontend so the prediction form can
        render the correct input widgets (number box vs dropdown).
    target_name : str
        Name of the target column (last column in the uploaded file).
    """
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    task: str
    input_dim: int
    output_dim: int
    preprocessor: ColumnTransformer
    label_encoder: LabelEncoder | None
    feature_columns: list[str]
    feature_schema: list[dict[str, Any]]
    target_name: str


# ---------------------------------------------------------------------------
# Helper functions (private, prefixed with underscore)
# ---------------------------------------------------------------------------

def _build_one_hot_encoder() -> OneHotEncoder:
    """
    Create a OneHotEncoder that works across different scikit-learn versions.

    Newer versions of sklearn (≥ 1.2) renamed the parameter from
    ``sparse`` to ``sparse_output``.  This function tries the new name
    first and falls back to the old name if a TypeError is raised.

    Parameters
    ----------
    (none)

    Returns
    -------
    OneHotEncoder
        Configured encoder that:
          - Outputs a dense (non-sparse) NumPy array (sparse_output=False).
          - Ignores unknown categories at transform time instead of raising
            an error (handle_unknown="ignore").  This is important because
            the prediction page might send a category value that was not
            present in the training set.
    """
    try:
        # Newer sklearn (≥ 1.2): parameter is called sparse_output
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Older sklearn (< 1.2): parameter is called sparse
        return OneHotEncoder(handle_unknown="ignore", sparse=False)



def _detect_task(y: pd.Series) -> str:
    """
    Automatically determine whether the prediction task is classification
    or regression by examining the target column values.

    Decision logic:
      1. If the column is non-numeric (e.g. strings like "Yes"/"No"),
         it is always classification.
      2. If it IS numeric but has 20 or fewer unique values, treat it as
         classification.  This handles cases like heart-disease datasets
         where the target is 0/1 stored as integers.
      3. Otherwise it is regression (continuous numeric output).

    Parameters
    ----------
    y : pd.Series
        The raw target column from the uploaded dataset, before any
        encoding or transformation.

    Returns
    -------
    str
        "classification" or "regression".
    """
    # Non-numeric dtype → always classification (e.g. string labels).
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"

    # Count distinct non-NaN values in the target column.
    unique_count = y.nunique(dropna=True)

    # Numeric column with few unique values → treat as categorical classes.
    # Threshold of 20 is a heuristic: datasets with 0/1 targets, or small
    # integer labels (0–9), are captured here.
    if unique_count <= 20:
        return "classification"

    # Many unique numeric values → continuous regression target.
    return "regression"



def _build_feature_schema(df_features: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Build a JSON-serializable metadata list describing every feature column.

    This schema serves two purposes:
      1. The prediction page (predict.html) reads it via /prediction_schema
         to dynamically create input fields – a number box for numeric
         features and a dropdown for categorical features.
      2. It records min/max/mean statistics that can be displayed as hints
         next to each input field (e.g. "range: 20–80, mean: 54.3").

    Parameters
    ----------
    df_features : pd.DataFrame
        The feature columns of the uploaded dataset (everything except the
        last target column).  Missing values may still be present.

    Returns
    -------
    list[dict[str, Any]]
        One dict per column.  Numeric columns contain keys:
          name, type("numeric"), min, max, mean, example.
        Categorical columns contain keys:
          name, type("categorical"), choices (sorted list), example.
    """
    # Initialize empty schema list – one entry will be added per column.
    schema: list[dict[str, Any]] = []

    # Iterate over every column in the feature DataFrame.
    for col in df_features.columns:
        if pd.api.types.is_numeric_dtype(df_features[col]):
            # ── Numeric column ──
            # Drop NaN values so min/max/mean calculations are accurate.
            col_data = df_features[col].dropna()

            # Pick the first valid value as an example default for the form.
            example_val = col_data.iloc[0] if len(col_data) > 0 else 0.0

            schema.append({
                "name": col,            # Column name string.
                "type": "numeric",      # Tells the frontend to render a number input.
                "min": float(col_data.min()),    # Minimum observed value.
                "max": float(col_data.max()),    # Maximum observed value.
                "mean": float(col_data.mean()),  # Mean – shown as a hint.
                "example": float(example_val),   # Pre-filled default value.
            })
        else:
            # ── Categorical column ──
            # Get all unique non-null values, cast to string, and sort them
            # alphabetically so the dropdown choices are in a predictable order.
            values = sorted(df_features[col].dropna().astype(str).unique().tolist())

            # Use the first sorted value as the default example.
            example_val = values[0] if values else ""

            schema.append({
                "name": col,                 # Column name string.
                "type": "categorical",       # Tells the frontend to render a <select>.
                "choices": values,           # All unique category values (sorted).
                "example": example_val,      # Default selection.
            })

    return schema


def _safe_classification_split(
    X,
    y,
    test_size: float,
    random_state: int,
):
    """
    Split features (X) and labels (y) into two parts for classification,
    attempting stratified splitting first and falling back to a simple
    random split if stratification fails.

    Stratified splitting ensures each split has roughly the same proportion
    of each class as the full dataset.  This can fail when a class has
    very few samples (e.g. only 1 sample), so a non-stratified fallback
    is provided to avoid crashing.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix to split.
    y : np.ndarray
        Integer-encoded target labels (from LabelEncoder).
    test_size : float
        Fraction of data to put in the second split (e.g. 0.2 = 20%).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    tuple
        (X_part1, X_part2, y_part1, y_part2) – the two splits.
    """
    try:
        # Attempt stratified split – preserves class proportions in each split.
        # stratify=y tells sklearn to use the label distribution as a guide.
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    except ValueError:
        # Fallback: some class might have too few samples to stratify.
        # Use a plain random split instead.
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def load_and_prepare_dataset(
    file_path: str,
    max_rows: int = MAX_ROWS_DEFAULT,
    random_state: int = 42,
) -> PreparedData:
    """
    Load a dataset from disk, preprocess it, and return a PreparedData object
    ready for neural network training.

    This is the single entry point called by app.py when a user uploads a
    file.  It orchestrates the entire data pipeline:

      1. Read the file (CSV or Excel) into a Pandas DataFrame.
      2. Validate that it has at least 2 columns (features + target).
      3. Down-sample to max_rows if the dataset is too large.
      4. Drop rows where the target value is missing.
      5. Separate features (all columns except last) from target (last column).
      6. Build the feature schema for the prediction UI.
      7. Identify numeric vs categorical feature columns.
      8. Construct sklearn preprocessing pipelines:
         - Numeric:      median imputation → standard scaling.
         - Categorical:  mode imputation → one-hot encoding.
      9. Combine them into a single ColumnTransformer.
     10. Detect the task type (classification or regression).
     11. For classification:
         a. Encode string labels to integers with LabelEncoder.
         b. Split data into train/val/test (60/20/20) with stratification.
         c. Fit the preprocessor on training data, transform all splits.
         d. One-hot encode the integer labels for Keras (to_categorical).
     12. For regression:
         a. Coerce target to float, drop rows that failed conversion.
         b. Split data into train/val/test (60/20/20).
         c. Fit the preprocessor on training data, transform all splits.
         d. Reshape target arrays to (n, 1).
     13. Return a PreparedData dataclass with all arrays and metadata.

    Parameters
    ----------
    file_path : str
        Absolute path to the uploaded dataset file on disk.
    max_rows : int, optional
        Maximum number of rows to use.  Larger datasets are randomly
        sampled down to this size.  Default is 20,000.
    random_state : int, optional
        Random seed for reproducibility of sampling and splitting.
        Default is 42.

    Returns
    -------
    PreparedData
        Fully preprocessed data ready for model training.

    Raises
    ------
    ValueError
        If the file format is unsupported, the dataset has fewer than 2
        columns, or no valid rows remain after cleaning.
    """

    # ── Step 1: Read the file into a Pandas DataFrame ──
    # Determine file type from extension and call the appropriate reader.
    if file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path)          # Read comma-separated values.
    elif file_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)         # Read Excel workbook.
    else:
        raise ValueError("Unsupported file format. Please upload CSV or Excel file.")

    # ── Step 2: Validate minimum column count ──
    # We need at least one feature column and one target column.
    if df.shape[1] < 2:
        raise ValueError("Dataset must contain at least one feature column and one target column.")

    # ── Step 3: Down-sample if the dataset exceeds max_rows ──
    # random_state ensures the same sample is taken each time for the same file.
    # reset_index(drop=True) re-numbers rows 0, 1, 2, … after sampling.
    if df.shape[0] > max_rows:
        df = df.sample(max_rows, random_state=random_state).reset_index(drop=True)

    # ── Step 4: Drop rows where the target (last column) is missing ──
    # Feature missing values will be handled later by the imputers, but a
    # missing target makes the row useless for supervised learning.
    target_name = df.columns[-1]  # The last column is the target by convention.
    df = df.dropna(subset=[target_name]).reset_index(drop=True)

    # ── Step 5: Separate features (X) from target (y) ──
    # X_df = all columns except the last one.
    # y    = just the last column.
    # .copy() ensures we own the data and avoid SettingWithCopyWarning.
    X_df = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    # Safety check: if all feature rows were dropped, we cannot proceed.
    if X_df.empty:
        raise ValueError("Feature matrix is empty after processing.")

    # ── Step 6: Build feature schema for the prediction form ──
    # This records metadata (type, min, max, choices, etc.) for each feature
    # column so the frontend can dynamically render input fields.
    feature_schema = _build_feature_schema(X_df)

    # ── Step 7: Identify numeric vs categorical columns ──
    # numeric_cols:     columns with numeric or boolean dtype.
    # categorical_cols: everything else (strings, object dtype, etc.).
    numeric_cols = X_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X_df.columns if c not in numeric_cols]

    # ── Step 8a: Build the numeric preprocessing pipeline ──
    # Step 1 ("imputer"): Replace missing numeric values with the column median.
    #   Median is robust to outliers (unlike mean).
    # Step 2 ("scaler"):  StandardScaler normalizes each column to have
    #   mean = 0 and standard deviation = 1.  This helps neural networks
    #   converge faster because all features are on the same scale.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # ── Step 8b: Build the categorical preprocessing pipeline ──
    # Step 1 ("imputer"): Replace missing categorical values with the most
    #   frequent (mode) value in that column.
    # Step 2 ("onehot"):  Convert each category into a set of binary columns.
    #   E.g. column "color" with values [red, blue] becomes two columns:
    #   color_red (0 or 1) and color_blue (0 or 1).
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _build_one_hot_encoder()),
        ]
    )

    # ── Step 9: Combine into a single ColumnTransformer ──
    # "num" transformer applies numeric_pipeline to numeric_cols.
    # "cat" transformer applies categorical_pipeline to categorical_cols.
    # remainder="drop" means any columns not listed are silently dropped
    # (there shouldn't be any, but this is a safety net).
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    # ── Step 10: Detect task type ──
    task = _detect_task(y)

    # =======================================================================
    # CLASSIFICATION PATH
    # =======================================================================
    if task == "classification":
        # Convert target values to strings so LabelEncoder treats every value
        # uniformly (even if the original column had mixed types like 0/1 ints).
        y = y.astype(str)

        # LabelEncoder maps each unique string label to a sequential integer:
        #   e.g. ["No", "Yes"] → [0, 1]
        # .fit_transform() learns the mapping AND transforms y in one step.
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)  # np.ndarray of ints

        # Number of distinct classes – this becomes the output_dim of the NN.
        n_classes = len(label_encoder.classes_)

        # ── Split 1: 80% train+val / 20% test ──
        # Uses _safe_classification_split which attempts stratified splitting
        # (preserving class ratios) and falls back to random if it fails.
        X_trainval_df, X_test_df, y_trainval, y_test = _safe_classification_split(
            X_df,
            y_encoded,
            test_size=0.2,        # 20% goes to test
            random_state=random_state,
        )

        # ── Split 2: 75% train / 25% val (of the 80% trainval) ──
        # 0.25 of 80% = 20% of the total → final ratio is 60/20/20.
        X_train_df, X_val_df, y_train, y_val = _safe_classification_split(
            X_trainval_df,
            y_trainval,
            test_size=0.25,       # 25% of 80% = 20% overall for validation
            random_state=random_state,
        )

        # ── Fit preprocessor on training data, transform all splits ──
        # fit_transform on train: learns medians, means, std devs, categories.
        # transform on val/test: applies the SAME learned parameters (no data leakage).
        X_train = preprocessor.fit_transform(X_train_df)  # np.ndarray (float64)
        X_val = preprocessor.transform(X_val_df)
        X_test = preprocessor.transform(X_test_df)

        # ── One-hot encode the integer labels for Keras ──
        # Keras CategoricalCrossentropy loss expects one-hot targets.
        # E.g. class index 1 with 3 classes → [0, 1, 0].
        y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
        y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes=n_classes)
        y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes=n_classes)

        # ── Package everything into PreparedData and return ──
        # .astype(np.float32) reduces memory and matches TensorFlow's default dtype.
        return PreparedData(
            X_train=X_train.astype(np.float32),
            X_val=X_val.astype(np.float32),
            X_test=X_test.astype(np.float32),
            y_train=y_train_oh.astype(np.float32),
            y_val=y_val_oh.astype(np.float32),
            y_test=y_test_oh.astype(np.float32),
            task=task,                               # "classification"
            input_dim=X_train.shape[1],              # Number of features after encoding
            output_dim=n_classes,                     # Number of output neurons
            preprocessor=preprocessor,               # Fitted ColumnTransformer
            label_encoder=label_encoder,             # For mapping predictions back to labels
            feature_columns=X_df.columns.tolist(),   # Original column names
            feature_schema=feature_schema,           # Metadata for the prediction form
            target_name=target_name,                 # Name of the target column
        )

    # =======================================================================
    # REGRESSION PATH
    # =======================================================================

    # Convert target to float; non-numeric strings become NaN via errors="coerce".
    y_float = pd.to_numeric(y, errors="coerce")

    # Create a boolean mask: True where the value is NOT NaN (i.e. valid).
    valid_mask = ~y_float.isna()

    # Keep only rows with valid numeric targets in both features and target.
    X_df = X_df.loc[valid_mask].reset_index(drop=True)
    y_float = y_float.loc[valid_mask].reset_index(drop=True)

    if X_df.empty:
        raise ValueError("No valid rows remain for regression target.")

    # ── Split 1: 80% train+val / 20% test (no stratification for regression) ──
    X_trainval_df, X_test_df, y_trainval, y_test = train_test_split(
        X_df,
        y_float.values,        # Convert Series to numpy array for splitting.
        test_size=0.2,
        random_state=random_state,
    )

    # ── Split 2: 75% train / 25% val of the 80% → 60/20/20 overall ──
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_trainval_df,
        y_trainval,
        test_size=0.25,
        random_state=random_state,
    )

    # ── Fit preprocessor on training data, transform all splits ──
    X_train = preprocessor.fit_transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)
    X_test = preprocessor.transform(X_test_df)

    # ── Package and return for regression ──
    # Target arrays are reshaped to (n, 1) because the NN output layer has
    # 1 neuron for regression, so Keras expects a 2-D target array.
    return PreparedData(
        X_train=X_train.astype(np.float32),
        X_val=X_val.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_train=np.asarray(y_train, dtype=np.float32).reshape(-1, 1),
        y_val=np.asarray(y_val, dtype=np.float32).reshape(-1, 1),
        y_test=np.asarray(y_test, dtype=np.float32).reshape(-1, 1),
        task=task,                               # "regression"
        input_dim=X_train.shape[1],              # Number of features after encoding
        output_dim=1,                            # Single regression output
        preprocessor=preprocessor,               # Fitted ColumnTransformer
        label_encoder=None,                      # Not used in regression
        feature_columns=X_df.columns.tolist(),   # Original column names
        feature_schema=feature_schema,           # Metadata for the prediction form
        target_name=target_name,                 # Name of the target column
    )
