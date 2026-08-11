from typing import Any
from pydantic import BaseModel, Field


class DatasetProfile(BaseModel):
    """Structured information summarizing the dataset characteristics."""

    row_count: int = Field(..., description="Number of rows in the dataset")
    column_count: int = Field(..., description="Number of columns in the dataset")
    columns: list[str] = Field(..., description="Ordered list of column names")
    dtypes: dict[str, str] = Field(..., description="Mapping of column names to their data types")
    missing_counts: dict[str, int] = Field(..., description="Count of missing values per column")
    numeric_columns: list[str] = Field(..., description="Columns detected as numeric")
    categorical_columns: list[str] = Field(..., description="Columns detected as categorical")
    summary_statistics: dict[str, dict[str, float]] = Field(
        ..., description="Descriptive statistics for numeric columns (mean, std, min, 25%, 50%, 75%, max)"
    )


class PreprocessingSummary(BaseModel):
    """Summary of transformation steps applied during preprocessing."""

    imputed_numeric_cols: list[str] = Field(default_factory=list, description="Numeric columns that underwent imputation")
    imputed_categorical_cols: list[str] = Field(default_factory=list, description="Categorical columns that underwent imputation")
    encoded_categorical_cols: list[str] = Field(default_factory=list, description="Categorical columns that were encoded (e.g. One-Hot)")
    scaled_numeric_cols: list[str] = Field(default_factory=list, description="Numeric columns that were scaled")


class MlMetrics(BaseModel):
    """Unified metrics structure supporting both classification and regression tasks."""

    # Classification metrics
    accuracy: float | None = Field(default=None, description="Accuracy score (classification only)")
    precision: float | None = Field(default=None, description="Precision score (classification only)")
    recall: float | None = Field(default=None, description="Recall score (classification only)")
    f1: float | None = Field(default=None, description="F1 score (classification only)")

    # Regression metrics
    mae: float | None = Field(default=None, description="Mean Absolute Error (regression only)")
    mse: float | None = Field(default=None, description="Mean Squared Error (regression only)")
    rmse: float | None = Field(default=None, description="Root Mean Squared Error (regression only)")
    r2: float | None = Field(default=None, description="Coefficient of determination R² (regression only)")


class FeatureImportance(BaseModel):
    """Unified representation of a single feature's relative importance score."""

    feature_name: str = Field(..., description="Name of the input feature")
    importance_score: float = Field(..., description="Importance score or model coefficient magnitude")


class TrainResponse(BaseModel):
    """Comprehensive details returned after training a model."""

    model_id: str = Field(..., description="Unique process-lifetime identifier of the trained model")
    dataset_summary: DatasetProfile = Field(..., description="Profile of the dataset used for training")
    task: str = Field(..., description="The supervised task resolved (classification or regression)")
    target: str = Field(..., description="Name of the target label column")
    preprocessing_summary: PreprocessingSummary = Field(..., description="Detailed preprocessing steps applied")
    model_name: str = Field(..., description="Class name of the trained estimator model")
    metrics: MlMetrics = Field(..., description="Evaluation metrics computed on test split split")
    feature_importance: list[FeatureImportance] = Field(..., description="Ranked features based on importance/coefficients")
    warnings: list[str] = Field(default_factory=list, description="Warnings generated during profiling, splitting, or training")


class TrainRequest(BaseModel):
    """Payload to request training of a machine learning model on custom data."""

    data: list[dict[str, Any]] = Field(..., description="List of dictionaries representing tabular data rows")
    target: str = Field(..., description="Name of the target column")
    task: str | None = Field(default=None, description="Supervised task: classification or regression (omitted for auto-detect)")
    model_type: str | None = Field(default=None, description="Specific model selection: e.g. LogisticRegression, RandomForestClassifier")


class PredictRequest(BaseModel):
    """Payload to query predictions from an existing trained model."""

    model_id: str = Field(..., description="UUID of the trained model in process-lifetime cache")
    data: list[dict[str, Any]] = Field(..., description="List of dictionaries representing input rows to predict")


class PredictResponse(BaseModel):
    """Structured responses containing prediction results and optional probability scores."""

    predictions: list[Any] = Field(..., description="List of predicted classes or continuous values")
    probabilities: list[list[float]] | None = Field(default=None, description="Probability scores for each class (classification only)")
