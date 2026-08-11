from typing import Any
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from poc_kanini.ml.models import PreprocessingSummary


class MlPreprocessor:
    """Encapsulates fit-transform preprocessing for tabular classification and regression."""

    def __init__(self, numeric_features: list[str], categorical_features: list[str]) -> None:
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.preprocessor: ColumnTransformer | None = None
        self._is_fit = False

        # Track what was actually processed for PreprocessingSummary metadata
        self.imputed_numeric_cols = []
        self.scaled_numeric_cols = []
        self.imputed_categorical_cols = []
        self.encoded_categorical_cols = []

    def fit(self, X: pd.DataFrame) -> "MlPreprocessor":
        """Fit the preprocessing transformers on the training dataframe.

        Args:
            X: Input training features DataFrame.
        """
        # Define numeric sub-pipeline
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        # Define categorical sub-pipeline
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        # Construct final column transformer
        transformers: list[tuple[str, Any, list[str]]] = []
        if self.numeric_features:
            transformers.append(("num", numeric_transformer, self.numeric_features))
        if self.categorical_features:
            transformers.append(("cat", categorical_transformer, self.categorical_features))

        self.preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        self.preprocessor.fit(X)
        self._is_fit = True

        # Populating preprocessing metadata summary
        if self.numeric_features:
            self.imputed_numeric_cols = list(self.numeric_features)
            self.scaled_numeric_cols = list(self.numeric_features)
        if self.categorical_features:
            self.imputed_categorical_cols = list(self.categorical_features)
            self.encoded_categorical_cols = list(self.categorical_features)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing transformations to a dataframe.

        Args:
            X: The input DataFrame.

        Returns:
            A preprocessed pandas DataFrame with transformed feature columns.
        """
        if not self._is_fit or self.preprocessor is None:
            raise RuntimeError("MlPreprocessor is not fitted yet. Call fit() before transform().")

        transformed_array = self.preprocessor.transform(X)
        feature_names = self.get_feature_names_out()

        # Convert back to clean DataFrame
        return pd.DataFrame(transformed_array, columns=feature_names, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit the transformers and transform the input dataframe.

        Args:
            X: Input training features DataFrame.

        Returns:
            Preprocessed features DataFrame.
        """
        return self.fit(X).transform(X)

    def get_feature_names_out(self) -> list[str]:
        """Get output feature names after preprocessing (including one-hot encoded columns)."""
        if not self._is_fit or self.preprocessor is None:
            return []

        # Scikit-learn ColumnTransformer has get_feature_names_out
        try:
            names = self.preprocessor.get_feature_names_out()
            # Clean up default prefix naming from ColumnTransformer (e.g. 'num__age' -> 'age')
            cleaned_names = []
            for name in names:
                if name.startswith("num__"):
                    cleaned_names.append(name[5:])
                elif name.startswith("cat__"):
                    cleaned_names.append(name[5:])
                else:
                    cleaned_names.append(name)
            return cleaned_names
        except Exception:
            # Fallback if names cannot be retrieved
            return [f"feature_{i}" for i in range(self.preprocessor.transform(X).shape[1])]

    def get_summary(self) -> PreprocessingSummary:
        """Return structured preprocessing summary metadata."""
        return PreprocessingSummary(
            imputed_numeric_cols=self.imputed_numeric_cols,
            imputed_categorical_cols=self.imputed_categorical_cols,
            encoded_categorical_cols=self.encoded_categorical_cols,
            scaled_numeric_cols=self.scaled_numeric_cols,
        )
