import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from poc_kanini.main import app
from poc_kanini.ml.models import DatasetProfile, TrainResponse, PredictResponse
from poc_kanini.ml.profiler import profile_dataframe
from poc_kanini.ml.preprocessor import MlPreprocessor
from poc_kanini.ml.engine import train_and_evaluate, MlEngineError
from poc_kanini.ml.service import MlService


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Generate a clean mock DataFrame for testing."""
    return pd.DataFrame(
        {
            "age": [25, 30, np.nan, 45, 50],
            "income": [50000.0, 60000.0, 75000.0, np.nan, 90000.0],
            "city": ["New York", "Chicago", "New York", "San Francisco", np.nan],
            "churned": [0, 1, 0, 1, 0],
        }
    )


@pytest.fixture
def classification_data() -> pd.DataFrame:
    """Generate a simple clean binary classification dataset."""
    np.random.seed(42)
    X1 = np.random.normal(loc=0.0, scale=1.0, size=20)
    X2 = np.random.normal(loc=1.0, scale=1.0, size=20)
    # Simple threshold target
    y = [1 if (x1 + x2) > 0.5 else 0 for x1, x2 in zip(X1, X2)]
    return pd.DataFrame({"feat1": X1, "feat2": X2, "target": y})


@pytest.fixture
def regression_data() -> pd.DataFrame:
    """Generate a simple clean regression dataset."""
    np.random.seed(42)
    X1 = np.random.normal(loc=5.0, scale=2.0, size=25)
    X2 = np.random.normal(loc=2.0, scale=1.0, size=25)
    # Simple linear relationship
    y = 2.5 * X1 - 1.2 * X2 + np.random.normal(0.0, 0.1, size=25)
    return pd.DataFrame({"feat1": X1, "feat2": X2, "target": y})


def test_profile_dataframe(sample_df):
    """Test that dataset profiling correctly computes metrics and detects columns."""
    profile = profile_dataframe(sample_df)

    assert isinstance(profile, DatasetProfile)
    assert profile.row_count == 5
    assert profile.column_count == 4
    assert set(profile.numeric_columns) == {"age", "income", "churned"}
    assert set(profile.categorical_columns) == {"city"}
    assert profile.missing_counts["age"] == 1
    assert profile.missing_counts["city"] == 1

    # Check statistics are present for numeric columns
    assert "age" in profile.summary_statistics
    assert profile.summary_statistics["age"]["mean"] == 37.5
    assert profile.summary_statistics["age"]["min"] == 25.0
    assert profile.summary_statistics["age"]["max"] == 50.0


def test_preprocessor(sample_df):
    """Test preprocessing imputation, scaling, and one-hot encoding."""
    X = sample_df[["age", "income", "city"]]
    numeric_features = ["age", "income"]
    categorical_features = ["city"]

    preprocessor = MlPreprocessor(numeric_features, categorical_features)
    X_processed = preprocessor.fit_transform(X)

    # SimpleImputer should fill missing values
    assert X_processed.isnull().sum().sum() == 0

    # StandardScaler should normalize numeric columns
    assert np.allclose(X_processed["age"].mean(), 0.0, atol=1e-5)

    # OneHotEncoder feature names output
    feature_names = preprocessor.get_feature_names_out()
    assert "city_New York" in feature_names
    assert "city_Chicago" in feature_names


def test_train_classification_logistic(classification_data):
    """Test training and evaluation of a Logistic Regression classifier."""
    model, preprocessor, response = train_and_evaluate(
        df=classification_data, target="target", task="classification", model_type="LogisticRegression"
    )

    assert isinstance(response, TrainResponse)
    assert response.task == "classification"
    assert response.model_name == "LogisticRegression"
    assert response.metrics.accuracy is not None
    assert response.metrics.f1 is not None
    assert len(response.feature_importance) == 2
    # Check that importance scores are absolute coefficients sorted descending
    assert response.feature_importance[0].importance_score >= response.feature_importance[1].importance_score


def test_train_classification_random_forest(classification_data):
    """Test training and evaluation of a Random Forest classifier."""
    model, preprocessor, response = train_and_evaluate(
        df=classification_data, target="target", task="classification", model_type="RandomForestClassifier"
    )

    assert isinstance(response, TrainResponse)
    assert response.task == "classification"
    assert response.model_name == "RandomForestClassifier"
    assert len(response.feature_importance) == 2
    assert response.feature_importance[0].importance_score >= response.feature_importance[1].importance_score


def test_train_regression_linear(regression_data):
    """Test training and evaluation of a Linear Regression model."""
    model, preprocessor, response = train_and_evaluate(
        df=regression_data, target="target", task="regression", model_type="LinearRegression"
    )

    assert isinstance(response, TrainResponse)
    assert response.task == "regression"
    assert response.model_name == "LinearRegression"
    assert response.metrics.mae is not None
    assert response.metrics.rmse is not None
    assert response.metrics.r2 is not None


def test_train_regression_random_forest(regression_data):
    """Test training and evaluation of a Random Forest Regressor."""
    model, preprocessor, response = train_and_evaluate(
        df=regression_data, target="target", task="regression", model_type="RandomForestRegressor"
    )

    assert isinstance(response, TrainResponse)
    assert response.task == "regression"
    assert response.model_name == "RandomForestRegressor"
    assert len(response.feature_importance) == 2


def test_invalid_inputs(sample_df):
    """Test ML Engine error handling for invalid configurations."""
    # Target column does not exist
    with pytest.raises(MlEngineError, match="does not exist"):
        train_and_evaluate(sample_df, target="non_existent_column")

    # Empty dataset
    empty_df = pd.DataFrame(columns=["age", "churned"])
    with pytest.raises(MlEngineError, match="Dataset is empty"):
        train_and_evaluate(empty_df, target="churned")

    # Target is completely null
    null_target_df = pd.DataFrame({"age": [20, 30], "target": [np.nan, np.nan]})
    with pytest.raises(MlEngineError, match="contains only missing values"):
        train_and_evaluate(null_target_df, target="target")

    # Classification task with fewer than 2 target classes
    single_class_df = pd.DataFrame({"age": [20, 30, 40], "target": [1, 1, 1]})
    with pytest.raises(MlEngineError, match="at least 2 distinct target classes"):
        train_and_evaluate(single_class_df, target="target", task="classification")


def test_ml_service_predict(classification_data):
    """Test end-to-end training and prediction caching in MlService."""
    service = MlService()
    records = classification_data.to_dict(orient="records")

    # Train
    train_res = service.train(data=records, target="target", task="classification")
    model_id = train_res.model_id
    assert model_id is not None

    # Predict
    test_records = [{"feat1": 0.5, "feat2": -0.2}, {"feat1": -1.2, "feat2": -1.5}]
    pred_res = service.predict(model_id=model_id, data=test_records)

    assert isinstance(pred_res, PredictResponse)
    assert len(pred_res.predictions) == 2
    assert pred_res.probabilities is not None
    assert len(pred_res.probabilities) == 2
    assert len(pred_res.probabilities[0]) == 2  # binary classification has 2 probabilities per row


def test_api_endpoints(classification_data):
    """Test the FastAPI endpoints using TestClient."""
    client = TestClient(app)
    records = classification_data.to_dict(orient="records")

    # 1. Profile Endpoint
    response = client.post("/api/ml/profile", json=records)
    assert response.status_code == 200
    profile_data = response.json()
    assert profile_data["row_count"] == 20
    assert "feat1" in profile_data["columns"]

    # 2. Train Endpoint
    train_payload = {
        "data": records,
        "target": "target",
        "task": "classification",
        "model_type": "LogisticRegression",
    }
    response = client.post("/api/ml/train", json=train_payload)
    assert response.status_code == 200
    train_data = response.json()
    assert "model_id" in train_data
    assert train_data["model_name"] == "LogisticRegression"
    assert train_data["task"] == "classification"
    model_id = train_data["model_id"]

    # 3. Predict Endpoint
    test_records = [{"feat1": 0.1, "feat2": 0.2}, {"feat1": -0.5, "feat2": -0.8}]
    predict_payload = {
        "model_id": model_id,
        "data": test_records,
    }
    response = client.post("/api/ml/predict", json=predict_payload)
    assert response.status_code == 200
    predict_data = response.json()
    assert len(predict_data["predictions"]) == 2
    assert predict_data["probabilities"] is not None


# ---------------------------------------------------------------------------
# Inline Dataset Parsing & Extraction Tests
# ---------------------------------------------------------------------------


def test_parse_inline_dataset_tuple_records() -> None:
    """Extract parenthesized records with header list from natural language prompt."""
    from poc_kanini.ml.dataset_parser import parse_inline_dataset

    prompt = (
        "Train a classification model using this dataset: feature1, feature2, churn. "
        "Use these records: (1,2,0), (2,3,0), (8,9,1), (9,10,0), (3,4,0), (7,8,1). "
        "Report the model metrics and feature importance."
    )
    records = parse_inline_dataset(prompt)
    assert isinstance(records, list)
    assert len(records) == 6
    assert records[0] == {"feature1": 1, "feature2": 2, "churn": 0}
    assert records[2] == {"feature1": 8, "feature2": 9, "churn": 1}


def test_parse_inline_dataset_json_list() -> None:
    """Extract JSON / Python literal list of dicts."""
    from poc_kanini.ml.dataset_parser import parse_inline_dataset

    prompt = "Profile this dataset [{'x': 10, 'y': 'A'}, {'x': 20, 'y': 'B'}, {'x': 30, 'y': 'A'}]"
    records = parse_inline_dataset(prompt)
    assert isinstance(records, list)
    assert len(records) == 3
    assert records[0] == {"x": 10, "y": "A"}


def test_parse_inline_dataset_multiline_csv() -> None:
    """Extract multiline CSV block from text."""
    from poc_kanini.ml.dataset_parser import parse_inline_dataset

    prompt = "Here is the data:\nfeature1,feature2,churn\n1,2,0\n2,3,0\n8,9,1\n9,10,1"
    records = parse_inline_dataset(prompt)
    assert isinstance(records, list)
    assert len(records) == 4
    assert records[0] == {"feature1": 1, "feature2": 2, "churn": 0}


def test_profile_dataset_tool_with_inline_dataset() -> None:
    """profile_dataset_tool accepts inline tuple-format string."""
    from poc_kanini.tools.data_tools import profile_dataset_tool

    prompt = "feature1, feature2, churn: (1,2,0), (2,3,0), (8,9,1), (9,10,0)"
    res = profile_dataset_tool.invoke({"data": prompt})
    assert "error" not in res
    assert res.get("row_count") == 4
    assert "churn" in res.get("columns", [])


def test_train_ml_model_tool_with_inline_dataset() -> None:
    """train_ml_model_tool accepts inline tuple-format string."""
    from poc_kanini.tools.ml_tools import train_ml_model_tool

    prompt = "feature1, feature2, churn: (1,2,0), (2,3,0), (8,9,1), (9,10,0), (3,4,0), (7,8,1)"
    res = train_ml_model_tool.invoke({"data": prompt, "target": "churn", "task": "classification"})
    assert "error" not in res
    assert "model_id" in res
    assert res.get("metrics", {}).get("accuracy") is not None


def test_train_ml_model_tool_no_dataset_honest_error() -> None:
    """train_ml_model_tool with empty/no dataset returns error without fabricating model."""
    from poc_kanini.tools.ml_tools import train_ml_model_tool

    res = train_ml_model_tool.invoke({"data": "", "target": "target"})
    assert "error" in res


def test_e2e_inline_dataset_hitl_approval_flow() -> None:
    """Test full flow: prompt with inline tuple dataset -> HITL approval -> ML training succeeds."""
    prompt = (
        "Train a classification model using this dataset: feature1, feature2, churn. "
        "Use these records: (1,2,0), (2,3,0), (8,9,1), (9,10,0), (3,4,0), (7,8,1). "
        "Report the model metrics and feature importance."
    )
    with TestClient(app) as client:
        # Turn 1: Post chat message with inline dataset
        r1 = client.post("/api/chat", json={"messages": [{"role": "user", "content": prompt}]})
        assert r1.status_code == 200
        b1 = r1.json()
        assert b1["approval_required"] is True
        thread_id = b1["thread_id"]
        appr_id = b1["approval_id"]

        # Turn 2: Approve the operation
        r2 = client.post(
            "/api/chat/approval",
            json={
                "thread_id": thread_id,
                "decision": "approved",
                "approval_id": appr_id,
            },
        )
        assert r2.status_code == 200
        b2 = r2.json()
        assert b2["approval_required"] is False

        # Verify that ML training executed on the inline dataset and has real metrics
        tool_results = b2.get("tool_results", [])
        ml_tool = next((t for t in tool_results if t.get("tool") == "train_ml_model_tool"), None)
        assert ml_tool is not None, f"train_ml_model_tool not in {tool_results}"
        assert "result" in ml_tool
        assert "model_id" in ml_tool["result"]
        assert ml_tool["result"]["model_id"] != "N/A"
        assert ml_tool["result"]["metrics"]["accuracy"] is not None


def test_e2e_ml_train_then_predict_multiturn() -> None:
    """Test full multi-turn workflow:
    Turn 1: Train model -> HITL required
    Turn 2: Approve -> model trained with model_id
    Turn 3: 'Use the model you just trained to predict the churn outcome for feature1=8 and feature2=9.'
            -> Uses existing model_id
            -> Calls predict_ml_model_tool
            -> Does NOT call train_ml_model_tool
            -> Returns prediction result without echoing duplicate training reports
    """
    train_prompt = (
        "Train a classification model using this dataset: feature1, feature2, churn. "
        "Use these records: (1,2,0), (2,3,0), (8,9,1), (9,10,0), (3,4,0), (7,8,1). "
        "Report the model metrics and feature importance."
    )
    with TestClient(app) as client:
        # Turn 1: Train
        r1 = client.post("/api/chat", json={"messages": [{"role": "user", "content": train_prompt}]})
        assert r1.status_code == 200
        b1 = r1.json()
        assert b1["approval_required"] is True
        thread_id = b1["thread_id"]
        appr_id = b1["approval_id"]

        # Turn 2: Approve
        r2 = client.post(
            "/api/chat/approval",
            json={
                "thread_id": thread_id,
                "decision": "approved",
                "approval_id": appr_id,
            },
        )
        assert r2.status_code == 200
        b2 = r2.json()
        assert b2["approval_required"] is False
        train_tool = next((t for t in b2.get("tool_results", []) if t.get("tool") == "train_ml_model_tool"), None)
        assert train_tool is not None
        trained_model_id = train_tool["result"]["model_id"]
        assert trained_model_id != "N/A"

        # Turn 3: Predict using the trained model
        predict_prompt = "Use the model you just trained to predict the churn outcome for feature1=8 and feature2=9."
        r3 = client.post(
            "/api/chat",
            json={
                "thread_id": thread_id,
                "messages": [{"role": "user", "content": predict_prompt}],
            },
        )
        assert r3.status_code == 200
        b3 = r3.json()
        assert b3["approval_required"] is False

        # Verify predict_ml_model_tool was executed
        tool_results_3 = b3.get("tool_results", [])
        pred_tool = next((t for t in tool_results_3 if t.get("tool") == "predict_ml_model_tool"), None)
        assert pred_tool is not None, f"predict_ml_model_tool missing from {tool_results_3}"
        assert "result" in pred_tool
        assert "predictions" in pred_tool["result"]
        assert len(pred_tool["result"]["predictions"]) == 1

        # Verify predict_ml_model_tool used the exact model_id from Turn 2
        assert pred_tool.get("model_id") == trained_model_id

        # Verify train_ml_model_tool was NOT called in Turn 3
        train_tool_3 = [t for t in tool_results_3 if t.get("tool") == "train_ml_model_tool" and t.get("query") == predict_prompt]
        assert len(train_tool_3) == 0, f"train_ml_model_tool was re-executed in Turn 3: {train_tool_3}"

        # Verify response text focuses on prediction result
        ans = b3.get("message", {}).get("content", "")
        assert "Prediction using model" in ans or "Predicted outcome" in ans or "Predictions:" in ans
        assert "Dataset contains" not in ans
        assert not ans.startswith("Trained LogisticRegression")


def test_e2e_ml_predict_without_model_returns_honest_error() -> None:
    """Test predict request when no model has been trained returns an honest error and does not silently train."""
    with TestClient(app) as client:
        r = client.post(
            "/api/chat",
            json={
                "messages": [
                    {"role": "user", "content": "Use the model you just trained to predict the churn outcome for feature1=8 and feature2=9."}
                ]
            },
        )
        assert r.status_code == 200
        body = r.json()
        # Should NOT trigger training or HITL
        assert body["approval_required"] is False
        ans = body.get("message", {}).get("content", "")
        assert "No trained model is available" in ans or "train a" in ans.lower()


