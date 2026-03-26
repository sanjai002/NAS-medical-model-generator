from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


MAX_ROWS_DEFAULT = 20_000


@dataclass
class PreparedData:
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



def _build_one_hot_encoder() -> OneHotEncoder:
    # sklearn compatibility across versions
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)



def _detect_task(y: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"

    unique_count = y.nunique(dropna=True)
    # Small-number numeric targets are handled as classes.
    if unique_count <= 20:
        return "classification"

    return "regression"



def _build_feature_schema(df_features: pd.DataFrame) -> list[dict[str, Any]]:
    schema: list[dict[str, Any]] = []
    for col in df_features.columns:
        if pd.api.types.is_numeric_dtype(df_features[col]):
            # Get min, max, mean for numeric fields
            col_data = df_features[col].dropna()
            example_val = col_data.iloc[0] if len(col_data) > 0 else 0.0
            schema.append({
                "name": col,
                "type": "numeric",
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "mean": float(col_data.mean()),
                "example": float(example_val),
            })
        else:
            values = sorted(df_features[col].dropna().astype(str).unique().tolist())
            example_val = values[0] if values else ""
            schema.append({
                "name": col,
                "type": "categorical",
                "choices": values,
                "example": example_val,
            })
    return schema


def _safe_classification_split(
    X,
    y,
    test_size: float,
    random_state: int,
):
    """Try stratified split first, then fallback to non-stratified for edge cases."""
    try:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    except ValueError:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )



def load_and_prepare_dataset(
    file_path: str,
    max_rows: int = MAX_ROWS_DEFAULT,
    random_state: int = 42,
) -> PreparedData:
    if file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload CSV or Excel file.")

    if df.shape[1] < 2:
        raise ValueError("Dataset must contain at least one feature column and one target column.")

    if df.shape[0] > max_rows:
        df = df.sample(max_rows, random_state=random_state).reset_index(drop=True)

    # Drop rows where target is missing; feature missing values are imputed.
    target_name = df.columns[-1]
    df = df.dropna(subset=[target_name]).reset_index(drop=True)

    X_df = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    if X_df.empty:
        raise ValueError("Feature matrix is empty after processing.")

    feature_schema = _build_feature_schema(X_df)

    numeric_cols = X_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X_df.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _build_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    task = _detect_task(y)

    if task == "classification":
        y = y.astype(str)
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        n_classes = len(label_encoder.classes_)

        X_trainval_df, X_test_df, y_trainval, y_test = _safe_classification_split(
            X_df,
            y_encoded,
            test_size=0.2,
            random_state=random_state,
        )

        X_train_df, X_val_df, y_train, y_val = _safe_classification_split(
            X_trainval_df,
            y_trainval,
            test_size=0.25,
            random_state=random_state,
        )

        X_train = preprocessor.fit_transform(X_train_df)
        X_val = preprocessor.transform(X_val_df)
        X_test = preprocessor.transform(X_test_df)

        y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
        y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes=n_classes)
        y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes=n_classes)

        return PreparedData(
            X_train=X_train.astype(np.float32),
            X_val=X_val.astype(np.float32),
            X_test=X_test.astype(np.float32),
            y_train=y_train_oh.astype(np.float32),
            y_val=y_val_oh.astype(np.float32),
            y_test=y_test_oh.astype(np.float32),
            task=task,
            input_dim=X_train.shape[1],
            output_dim=n_classes,
            preprocessor=preprocessor,
            label_encoder=label_encoder,
            feature_columns=X_df.columns.tolist(),
            feature_schema=feature_schema,
            target_name=target_name,
        )

    # regression
    y_float = pd.to_numeric(y, errors="coerce")
    valid_mask = ~y_float.isna()
    X_df = X_df.loc[valid_mask].reset_index(drop=True)
    y_float = y_float.loc[valid_mask].reset_index(drop=True)

    if X_df.empty:
        raise ValueError("No valid rows remain for regression target.")

    X_trainval_df, X_test_df, y_trainval, y_test = train_test_split(
        X_df,
        y_float.values,
        test_size=0.2,
        random_state=random_state,
    )

    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_trainval_df,
        y_trainval,
        test_size=0.25,
        random_state=random_state,
    )

    X_train = preprocessor.fit_transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)
    X_test = preprocessor.transform(X_test_df)

    return PreparedData(
        X_train=X_train.astype(np.float32),
        X_val=X_val.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_train=np.asarray(y_train, dtype=np.float32).reshape(-1, 1),
        y_val=np.asarray(y_val, dtype=np.float32).reshape(-1, 1),
        y_test=np.asarray(y_test, dtype=np.float32).reshape(-1, 1),
        task=task,
        input_dim=X_train.shape[1],
        output_dim=1,
        preprocessor=preprocessor,
        label_encoder=None,
        feature_columns=X_df.columns.tolist(),
        feature_schema=feature_schema,
        target_name=target_name,
    )
