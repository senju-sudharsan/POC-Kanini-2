from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from poc_kanini.ml.models import (
    FeatureImportance,
    MlMetrics,
    PreprocessingSummary,
    TrainResponse,
)
from poc_kanini.ml.preprocessor import MlPreprocessor
from poc_kanini.ml.profiler import profile_dataframe


class MlEngineError(ValueError):
    """Custom exception class for structured validation errors in the ML Engine."""
    pass


def train_and_evaluate(
    df: pd.DataFrame,
    target: str,
    task: str | None = None,
    model_type: str | None = None,
    model_id: str = "temp-id",
) -> tuple[Any, MlPreprocessor, TrainResponse]:
    """Train a model on the provided dataset and evaluate its metrics.

    Args:
        df: Input DataFrame containing features and target column.
        target: Name of the target label column.
        task: Optional task type ('classification' or 'regression').
        model_type: Optional model type to train.
        model_id: Process-lifetime UUID to assign to the trained pipeline.

    Returns:
        A tuple of (fitted_model, fitted_preprocessor, TrainResponse).
    """
    warnings = []

    # 1. Validation checks
    if target not in df.columns:
        raise MlEngineError(f"Target column '{target}' does not exist in the dataset.")

    if len(df) == 0:
        raise MlEngineError("Dataset is empty. Cannot train a model.")

    # Target missing value check
    if df[target].isnull().all():
        raise MlEngineError(f"Target column '{target}' contains only missing values.")

    # Drop target null rows and warn
    null_target_count = int(df[target].isnull().sum())
    if null_target_count > 0:
        warnings.append(
            f"Target column '{target}' contained {null_target_count} missing values. "
            "These rows were omitted from training."
        )
        df_clean = df.dropna(subset=[target])
    else:
        df_clean = df.copy()

    # Re-check rows after target null cleanup
    if len(df_clean) == 0:
        raise MlEngineError("No training samples left after removing rows with missing target values.")

    # 2. Extract features and target
    feature_cols = [col for col in df_clean.columns if col != target]
    X = df_clean[feature_cols]
    y = df_clean[target]

    # 3. Detect task if not specified
    if task is None:
        is_float = pd.api.types.is_float_dtype(y)
        cardinality = y.nunique()
        if is_float or (pd.api.types.is_numeric_dtype(y) and cardinality > 15):
            task = "regression"
        else:
            task = "classification"
    elif task not in ["classification", "regression"]:
        raise MlEngineError(f"Unsupported task type '{task}'. Use 'classification' or 'regression'.")

    # Double check target constraints based on task
    if task == "classification":
        unique_targets = y.nunique()
        if unique_targets < 2:
            raise MlEngineError(
                f"Classification task requires at least 2 distinct target classes, but found {unique_targets} class(es)."
            )
        if unique_targets > 50:
            warnings.append(
                f"High cardinality detected for classification target: {unique_targets} classes. "
                "Ensure this is correct."
            )

    # 4. Select baseline model
    model_name = ""
    model = None
    if task == "classification":
        if model_type is None:
            model_type = "LogisticRegression"

        if model_type == "LogisticRegression":
            model = LogisticRegression(max_iter=1000, random_state=42)
            model_name = "LogisticRegression"
        elif model_type == "RandomForestClassifier":
            model = RandomForestClassifier(random_state=42)
            model_name = "RandomForestClassifier"
        else:
            raise MlEngineError(
                f"Unsupported model type '{model_type}' for classification. "
                "Supported: LogisticRegression, RandomForestClassifier."
            )
    else:  # regression
        if model_type is None:
            model_type = "LinearRegression"

        if model_type == "LinearRegression":
            model = LinearRegression()
            model_name = "LinearRegression"
        elif model_type == "RandomForestRegressor":
            model = RandomForestRegressor(random_state=42)
            model_name = "RandomForestRegressor"
        else:
            raise MlEngineError(
                f"Unsupported model type '{model_type}' for regression. "
                "Supported: LinearRegression, RandomForestRegressor."
            )

    # 5. Split train/test
    if len(df_clean) < 10:
        warnings.append("Dataset is very small (< 10 rows). Train/test split metrics may not be stable.")
        if len(df_clean) < 4:
            warnings.append("Dataset is too small for split validation. Training and evaluating on the same data.")
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    else:
        stratify = None
        if task == "classification":
            class_counts = y.value_counts()
            if (class_counts >= 2).all() and len(class_counts) > 1:
                stratify = y
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify
            )
        except Exception:
            # Fallback to non-stratified split if stratification fails
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

    # 6. Fit preprocessing on training features
    numeric_features = [str(c) for c in X.select_dtypes(include=[np.number]).columns]
    categorical_features = [str(c) for c in X.columns if str(c) not in numeric_features]

    preprocessor = MlPreprocessor(numeric_features, categorical_features)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 7. Model fitting
    model.fit(X_train_processed, y_train)

    # 8. Evaluation
    y_pred = model.predict(X_test_processed)

    metrics = MlMetrics()
    if task == "classification":
        metrics.accuracy = float(accuracy_score(y_test, y_pred))
        # Handle multiclass metrics
        average_mode = "binary" if y.nunique() == 2 else "macro"
        metrics.precision = float(precision_score(y_test, y_pred, average=average_mode, zero_division=0))
        metrics.recall = float(recall_score(y_test, y_pred, average=average_mode, zero_division=0))
        metrics.f1 = float(f1_score(y_test, y_pred, average=average_mode, zero_division=0))
    else:  # regression
        metrics.mae = float(mean_absolute_error(y_test, y_pred))
        mse_val = mean_squared_error(y_test, y_pred)
        metrics.mse = float(mse_val)
        metrics.rmse = float(np.sqrt(mse_val))
        metrics.r2 = float(r2_score(y_test, y_pred))

    # 9. Explainability / Feature Importance
    feature_names = preprocessor.get_feature_names_out()
    feature_importances: list[FeatureImportance] = []

    if hasattr(model, "coef_"):
        coefs = model.coef_
        if len(coefs.shape) > 1:
            # Multiclass coefficients - take average absolute value across classes
            scores = np.mean(np.abs(coefs), axis=0)
        else:
            # Binary / single-target coefficients - take absolute value
            scores = np.abs(coefs)

        for name, score in zip(feature_names, scores):
            feature_importances.append(FeatureImportance(feature_name=name, importance_score=float(score)))
    elif hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
        for name, score in zip(feature_names, scores):
            feature_importances.append(FeatureImportance(feature_name=name, importance_score=float(score)))
    else:
        # Default fallback
        for name in feature_names:
            feature_importances.append(FeatureImportance(feature_name=name, importance_score=0.0))

    # Sort feature importances descending
    feature_importances.sort(key=lambda x: x.importance_score, reverse=True)

    # 10. Generate dataset profile
    dataset_summary = profile_dataframe(df)

    # 11. Build final response
    response = TrainResponse(
        model_id=model_id,
        dataset_summary=dataset_summary,
        task=task,
        target=target,
        preprocessing_summary=preprocessor.get_summary(),
        model_name=model_name,
        metrics=metrics,
        feature_importance=feature_importances,
        warnings=warnings,
    )

    return model, preprocessor, response
