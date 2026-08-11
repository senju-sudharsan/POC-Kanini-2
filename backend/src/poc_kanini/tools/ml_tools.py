"""Machine Learning model training and prediction tools wrapping MlService."""

import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from poc_kanini.ml.service import MlService

logger = logging.getLogger(__name__)

# Shared process-lifetime MlService instance for model caching across tool calls
ml_service_instance = MlService()

# Mapping convenient user/LLM aliases to engine's canonical class names
MODEL_TYPE_ALIASES = {
    "logistic": "LogisticRegression",
    "logisticregression": "LogisticRegression",
    "random_forest_classifier": "RandomForestClassifier",
    "randomforestclassifier": "RandomForestClassifier",
    "linear": "LinearRegression",
    "linearregression": "LinearRegression",
    "random_forest_regressor": "RandomForestRegressor",
    "randomforestregressor": "RandomForestRegressor",
}


class MlTrainInput(BaseModel):
    """Input schema for machine learning training tool."""

    data: list[dict[str, Any]] | str = Field(
        ...,
        description=(
            "Training dataset. Pass a list of record dicts or raw CSV string content. "
            "Arbitrary filesystem paths are NOT accepted for security."
        ),
    )
    target: str = Field(
        ...,
        description="Name of the target column to predict.",
    )
    task: str | None = Field(
        default=None,
        description="Task type: 'classification' or 'regression'. Auto-detected if omitted.",
    )
    model_type: str | None = Field(
        default=None,
        description=(
            "Model algorithm: 'LogisticRegression' or 'RandomForestClassifier' for classification; "
            "'LinearRegression' or 'RandomForestRegressor' for regression."
        ),
    )


class MlPredictInput(BaseModel):
    """Input schema for machine learning prediction tool."""

    model_id: str = Field(
        ...,
        description="UUID of a previously trained model held in process-lifetime cache.",
    )
    data: list[dict[str, Any]] | str = Field(
        ...,
        description="New feature records for prediction (list of dicts or CSV string).",
    )


def _normalize_model_type(model_type: str | None, task: str | None) -> str | None:
    """Normalize model_type aliases to canonical engine model names."""
    if model_type is None:
        return None
    normalized_key = model_type.casefold().replace(" ", "_").replace("-", "_")
    if normalized_key in MODEL_TYPE_ALIASES:
        return MODEL_TYPE_ALIASES[normalized_key]
    if normalized_key == "random_forest":
        if task == "regression":
            return "RandomForestRegressor"
        return "RandomForestClassifier"
    return model_type


@tool("train_ml_model_tool", args_schema=MlTrainInput)
def train_ml_model_tool(
    data: list[dict[str, Any]] | str,
    target: str,
    task: str | None = None,
    model_type: str | None = None,
) -> dict[str, Any]:
    """Train a supervised Machine Learning model (classification or regression) on a tabular dataset.

    Returns evaluation metrics (accuracy/F1 or MAE/RMSE/R²), feature importance,
    and a UUID model_id. The fitted model is stored in a process-lifetime cache for subsequent prediction tool calls.
    """
    normalized_model_type = _normalize_model_type(model_type, task)
    try:
        response = ml_service_instance.train(
            data=data,
            target=target,
            task=task,
            model_type=normalized_model_type,
        )
        return response.model_dump()
    except ValueError as error:
        return {"error": str(error)}
    except Exception as error:
        logger.error("Error in train_ml_model_tool: %s", error)
        return {"error": f"Model training failed: {error}"}


@tool("predict_ml_model_tool", args_schema=MlPredictInput)
def predict_ml_model_tool(
    model_id: str,
    data: list[dict[str, Any]] | str,
) -> dict[str, Any]:
    """Generate model predictions using a previously trained model ID.

    Note: Models are stored in an in-memory process-lifetime cache and will disappear
    if the backend process restarts.
    """
    try:
        response = ml_service_instance.predict(
            model_id=model_id,
            data=data,
        )
        res_dict = response.model_dump()
        res_dict["model_id"] = model_id
        return res_dict
    except ValueError as error:
        return {"error": str(error)}
    except Exception as error:
        logger.error("Error in predict_ml_model_tool: %s", error)
        return {"error": f"Model prediction failed: {error}"}
