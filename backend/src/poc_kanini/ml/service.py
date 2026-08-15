import io
import threading
import uuid
from typing import Any
import pandas as pd

from poc_kanini.ml.engine import train_and_evaluate
from poc_kanini.ml.models import (
    DatasetProfile,
    PredictResponse,
    TrainResponse,
)
from poc_kanini.ml.preprocessor import MlPreprocessor
from poc_kanini.ml.profiler import profile_dataframe


class MlService:
    """Orchestrates Machine Learning datasets profiling, training, and evaluation.

    This service implements a thread-safe, process-lifetime model cache.
    To support persistent database or cloud model storage later, override
    `_save_pipeline` and `_load_pipeline` methods.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # In-memory storage of fitted pipelines: model_id -> (model, preprocessor, task, target)
        self._pipelines: dict[str, tuple[Any, MlPreprocessor, str, str]] = {}

    def _save_pipeline(
        self, model_id: str, model: Any, preprocessor: MlPreprocessor, task: str, target: str
    ) -> None:
        """Saves a trained pipeline to storage. Default is in-memory process cache."""
        with self._lock:
            self._pipelines[model_id] = (model, preprocessor, task, target)

    def _load_pipeline(self, model_id: str) -> tuple[Any, MlPreprocessor, str, str]:
        """Loads a trained pipeline from storage. Default is in-memory process cache.

        Raises:
            KeyError: If model_id is not found.
        """
        with self._lock:
            if model_id not in self._pipelines:
                raise KeyError(f"Model ID '{model_id}' not found in model storage cache.")
            return self._pipelines[model_id]

    def _parse_data(self, data: list[dict[str, Any]] | str) -> pd.DataFrame:
        """Parse input data (JSON records list or CSV text string) into a pandas DataFrame."""
        from poc_kanini.ml.dataset_parser import parse_data_to_dataframe
        return parse_data_to_dataframe(data)

    def profile(self, data: list[dict[str, Any]] | str) -> DatasetProfile:
        """Parse input data and extract structured dataset characteristics."""
        df = self._parse_data(data)
        if df.empty:
            raise ValueError("Input dataset is empty.")
        return profile_dataframe(df)

    def train(
        self,
        data: list[dict[str, Any]] | str,
        target: str,
        task: str | None = None,
        model_type: str | None = None,
    ) -> TrainResponse:
        """Profile dataset, train a model, compute metrics, and cache the fitted pipeline."""
        df = self._parse_data(data)
        model_id = str(uuid.uuid4())

        # Fit model and evaluate
        model, preprocessor, response = train_and_evaluate(
            df=df, target=target, task=task, model_type=model_type, model_id=model_id
        )

        # Cache the fitted pipeline
        self._save_pipeline(model_id, model, preprocessor, response.task, target)

        return response

    def predict(self, model_id: str, data: list[dict[str, Any]] | str) -> PredictResponse:
        """Run predictions on new records using a cached trained model."""
        # Load trained pipeline
        try:
            model, preprocessor, task, target = self._load_pipeline(model_id)
        except KeyError as error:
            raise ValueError(str(error)) from error

        df = self._parse_data(data)
        if df.empty:
            raise ValueError("No prediction input records provided.")

        # Ensure features don't include the target column if it happens to be present
        feature_cols = [col for col in df.columns if col != target]
        X = df[feature_cols]

        # Ensure all numeric/categorical features expected by the preprocessor exist.
        # If columns are missing, align/impute them or trigger warnings
        expected_features = preprocessor.numeric_features + preprocessor.categorical_features
        for feature in expected_features:
            if feature not in X.columns:
                # Add empty column so SimpleImputer can handle it
                X[feature] = None

        # Preprocess features
        X_processed = preprocessor.transform(X)

        # Run predictions
        predictions = model.predict(X_processed).tolist()

        # Capture prediction probabilities if classification model supports it
        probabilities = None
        if task == "classification" and hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_processed).tolist()

        return PredictResponse(predictions=predictions, probabilities=probabilities)
