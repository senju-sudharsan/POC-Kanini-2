"""Comprehensive tests for Phase A General-Purpose Data Visualization engine and graph integration."""

import io
import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient

from poc_kanini.main import app
from poc_kanini.services.visualization_service import VisualizationService
from poc_kanini.tools.visualization_tools import visualize_dataset_tool

# Sample Multi-Domain Datasets
DATASET_1_SALES = (
    "product,revenue,units_sold\n"
    "Laptop,45000,30\n"
    "Smartphone,28000,70\n"
    "Headphones,6000,120\n"
    "Monitor,15000,45\n"
)

DATASET_2_HR = (
    "department,employees,avg_salary\n"
    "Engineering,85,125000\n"
    "Marketing,24,82000\n"
    "Sales,42,91000\n"
    "HR,12,74000\n"
    "Design,16,88000\n"
)

DATASET_3_WEATHER = (
    "date,temperature,humidity\n"
    "2026-01-01,18.5,65\n"
    "2026-01-02,19.2,60\n"
    "2026-01-03,21.0,58\n"
    "2026-01-04,17.8,70\n"
    "2026-01-05,22.4,54\n"
)

DATASET_4_HEALTH = (
    "patient_id,height_cm,weight_kg\n"
    "P001,175,72.5\n"
    "P002,162,58.0\n"
    "P003,188,91.2\n"
    "P004,170,68.4\n"
    "P005,155,51.0\n"
)

DATASET_5_TRANSPORT = (
    "vehicle_type,fuel_efficiency,trips\n"
    "Electric,110,450\n"
    "Hybrid,52,620\n"
    "Diesel,34,890\n"
    "Gasoline,28,1100\n"
)


def test_multi_dataset_dynamic_grounding_same_query():
    """Verify that identical queries on different datasets produce completely distinct, grounded outputs."""
    service = VisualizationService()
    generic_query = "plot the data as a chart"

    # Dataset 1: Sales
    res1 = service.visualize(DATASET_1_SALES, query=generic_query)
    assert len(res1) == 1
    v1 = res1[0]
    assert v1["chart_type"] == "bar"
    assert v1["x_field"] == "product"
    assert v1["y_field"] == "revenue"
    # Verify exact values derived from Dataset 1
    data1_map = {d["label"]: d["value"] for d in v1["data"]}
    assert data1_map["Laptop"] == 45000
    assert data1_map["Smartphone"] == 28000

    # Dataset 2: HR
    res2 = service.visualize(DATASET_2_HR, query=generic_query)
    assert len(res2) == 1
    v2 = res2[0]
    assert v2["chart_type"] == "bar"
    assert v2["x_field"] == "department"
    assert v2["y_field"] == "employees"
    data2_map = {d["label"]: d["value"] for d in v2["data"]}
    assert data2_map["Engineering"] == 85
    assert data2_map["HR"] == 12

    # Dataset 3: Weather (time series detected -> line chart)
    res3 = service.visualize(DATASET_3_WEATHER, query=generic_query)
    assert len(res3) == 1
    v3 = res3[0]
    assert v3["chart_type"] == "line"
    assert v3["x_field"] == "date"
    assert v3["y_field"] == "temperature"
    data3_vals = [d["value"] for d in v3["data"]]
    assert data3_vals == [18.5, 19.2, 21.0, 17.8, 22.4]

    # Dataset 4: Health (scatter plot query)
    res4 = service.visualize(DATASET_4_HEALTH, query="scatter plot of height_cm vs weight_kg")
    assert len(res4) == 1
    v4 = res4[0]
    assert v4["chart_type"] == "scatter"
    assert v4["x_field"] == "height_cm"
    assert v4["y_field"] == "weight_kg"
    assert len(v4["data"]) == 5
    assert v4["data"][0] == {"x": 175.0, "y": 72.5, "label": "(175, 72.5)"}


def test_pie_and_donut_charts():
    """Verify pie/donut charts are constructed with percentages and category groupings."""
    service = VisualizationService()
    res = service.visualize(DATASET_5_TRANSPORT, query="pie chart of trips by vehicle_type")
    assert len(res) == 1
    v = res[0]
    assert v["chart_type"] == "pie"
    assert v["x_field"] == "vehicle_type"
    assert v["y_field"] == "trips"
    # Check that highest trip category is at the top
    assert v["data"][0]["label"] == "Gasoline"
    assert v["data"][0]["value"] == 1100


def test_kpi_summary_metrics():
    """Verify KPI card generation with accurate record count and numerical aggregations."""
    service = VisualizationService()
    res = service.visualize(DATASET_2_HR, query="show key performance metrics and KPIs")
    assert len(res) == 1
    v = res[0]
    assert v["chart_type"] == "kpi"
    assert len(v["kpis"]) >= 3
    # First KPI is total records
    assert v["kpis"][0]["label"] == "Total Records"
    assert v["kpis"][0]["value"] == 5
    # Total employees sum = 85 + 24 + 42 + 12 + 16 = 179
    emp_kpi = next(k for k in v["kpis"] if "employees" in k["label"].lower())
    assert emp_kpi["value"] == "179" or emp_kpi["value"] == 179


def test_multi_chart_generation():
    """Verify generating multiple diverse charts when asked for '3 useful visualizations'."""
    service = VisualizationService()
    res = service.visualize(DATASET_1_SALES, query="Give me 3 useful visualizations from this dataset")
    assert len(res) == 3
    chart_types = [spec["chart_type"] for spec in res]
    assert "kpi" in chart_types
    assert "bar" in chart_types


def test_honest_failure_on_missing_column():
    """Verify that asking for non-existent columns returns an honest descriptive error instead of fabricating data."""
    service = VisualizationService()
    # Asking for "profit by region" on a dataset containing department,employees,avg_salary
    res = service.visualize(DATASET_2_HR, query="plot profit by region")
    assert len(res) == 1
    v = res[0]
    assert v["error"] is not None
    assert "does not contain field" in v["error"]
    assert "profit" in v["error"] or "region" in v["error"]
    assert "department" in v["error"]


def test_empty_dataset_handling():
    """Verify empty dataset returns clean error spec."""
    service = VisualizationService()
    res = service.visualize("", query="create a bar chart")
    assert len(res) == 1
    assert res[0]["error"] is not None


def test_tool_wrapper_execution():
    """Verify visualize_dataset_tool executes cleanly and returns structured payload."""
    out = visualize_dataset_tool.invoke({
        "data": DATASET_1_SALES,
        "query": "bar chart of revenue by product",
    })
    assert "visualizations" in out
    assert out["count"] == 1
    v = out["visualizations"][0]
    assert v["chart_type"] == "bar"
    assert v["title"] == "Total revenue by product"


from fastapi.testclient import TestClient


def test_e2e_chat_visualization_with_csv_data():
    """Verify end-to-end /api/chat endpoint with CSV attachment returns grounded visualizations."""
    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "Show me a bar chart of average salary by department"}],
                "csv_data": DATASET_2_HR,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert "visualizations" in payload
        assert len(payload["visualizations"]) >= 1
        viz = payload["visualizations"][0]
        assert viz["chart_type"] == "bar"
        assert viz["x_field"] == "department"
        assert viz["y_field"] == "avg_salary"


def test_visualization_turn_isolation():
    """Verify that visualizations generated in Turn 1 do not leak into Turn 2's response."""
    with TestClient(app) as client:
        # Turn 1: Chart request
        res1 = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "Visualize revenue by product"}],
                "csv_data": DATASET_1_SALES,
            },
        )
        assert res1.status_code == 200
        data1 = res1.json()
        thread_id = data1["thread_id"]
        assert len(data1.get("visualizations", [])) >= 1

        # Turn 2: Non-chart conversation in same thread
        res2 = client.post(
            "/api/chat",
            json={
                "thread_id": thread_id,
                "messages": [{"role": "user", "content": "What is the capital of France?"}],
            },
        )
        assert res2.status_code == 200
        data2 = res2.json()
        # Turn 2 must NOT have visualizations from Turn 1
        assert len(data2.get("visualizations", [])) == 0


# ===========================================================================
# Phase A Live Routing Regression Tests
# ===========================================================================

EXACT_OLIST_MULTI_CHART_QUERY = (
    "I want to test AURA's data visualization capability. Using the available Olist e-commerce dataset, "
    "create 3 different useful visualizations based on the actual dataset values. "
    "A bar chart comparing order counts by order_status. "
    "A line chart showing order volume over time using order_purchase_timestamp, aggregated by month. "
    "A scatter chart showing the relationship between two relevant numeric variables from the dataset. "
    "Choose the most meaningful pair based on the available columns and explain why you chose them. "
    "Important: Actually generate the visualization artifacts for AURA's frontend to render. "
    "Do not merely describe recommended charts or return chart specifications as plain text. "
    "Use only values computed from the actual dataset. Do not fabricate, estimate, or substitute sample data. "
    "Make the charts visually polished, readable, responsive, and appropriate for AURA's dark UI. "
    "If one requested chart cannot be produced because the necessary data/columns are unavailable, "
    "honestly explain why and produce the other valid visualizations instead. "
    "Do not use a generic fallback pretending a chart was created."
)

SAMPLE_OLIST_CSV = (
    "order_id,customer_id,order_status,order_purchase_timestamp,payment_value,freight_value\n"
    "ord_001,cust_001,delivered,2023-01-15 10:00:00,125.50,15.20\n"
    "ord_002,cust_002,shipped,2023-01-18 14:30:00,89.00,12.00\n"
    "ord_003,cust_003,delivered,2023-02-05 09:15:00,240.00,28.50\n"
    "ord_004,cust_004,canceled,2023-02-20 16:45:00,45.00,8.00\n"
    "ord_005,cust_005,delivered,2023-03-01 11:20:00,310.00,35.00\n"
    "ord_006,cust_006,delivered,2023-03-12 18:00:00,195.00,22.00\n"
)


def test_supervisor_routes_simple_chart_to_data():
    """'Create a bar chart of orders by order_status' must route to data, not ML."""
    from poc_kanini.graphs.supervisor import SupervisorRouter
    router = SupervisorRouter()
    decision = router._deterministic_route("Create a bar chart of orders by order_status")
    assert decision is not None
    assert decision.route == "data"


def test_supervisor_routes_exact_olist_query_to_data():
    """Exact multi-chart user prompt must route to data specialist."""
    from poc_kanini.graphs.supervisor import SupervisorRouter
    router = SupervisorRouter()
    decision = router._deterministic_route(EXACT_OLIST_MULTI_CHART_QUERY)
    assert decision is not None
    assert decision.route == "data"


def test_visualization_request_does_not_invoke_ml_or_require_approval():
    """Visualization request must invoke visualize_dataset_tool and NEVER trigger ML training or HITL approval."""
    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": EXACT_OLIST_MULTI_CHART_QUERY}],
                "csv_data": SAMPLE_OLIST_CSV,
            },
        )
        assert response.status_code == 200
        payload = response.json()

        # Must NOT require human approval for ML
        assert payload["approval_required"] is False
        assert payload.get("approval_id") is None
        assert payload.get("operation") is None

        # Must NOT have run train_ml_model_tool
        tools_run = [t.get("tool") for t in payload.get("tool_results", [])]
        assert "train_ml_model_tool" not in tools_run
        assert "visualize_dataset_tool" in tools_run

        # Must have returned actual computed visualizations
        assert len(payload.get("visualizations", [])) >= 1
        v = payload["visualizations"][0]
        assert v.get("chart_type") in ("bar", "line", "scatter", "kpi", "table", "pie", "donut")


def test_explicit_ml_training_still_routes_to_ml_and_requires_approval():
    """'Train a classification model...' must still route to ML and trigger HITL approval."""
    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "Train a classification model to predict order_status"}],
                "csv_data": SAMPLE_OLIST_CSV,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["approval_required"] is True
        assert payload.get("operation") == "ml"
        tools_run = [t.get("tool") for t in payload.get("tool_results", [])]
        assert "train_ml_model_tool" in tools_run


def test_explicit_ml_prediction_still_routes_to_ml():
    """'Use the model to predict...' must still route to ML prediction."""
    from poc_kanini.graphs.supervisor import SupervisorRouter
    router = SupervisorRouter()
    decision = router._deterministic_route("Use the trained model to predict churn for features payment_value=100")
    assert decision is not None
    assert decision.route == "ml"


def test_mixed_dataset_and_visualization_query_prioritizes_data_route():
    """A query mentioning dataset terms and chart terms prioritizes data route when no ML training is requested."""
    from poc_kanini.graphs.supervisor import SupervisorRouter
    router = SupervisorRouter()
    decision = router._deterministic_route("Using the uploaded tabular dataset with columns, show me a chart and summary metrics")
    assert decision is not None
    assert decision.route == "data"


# ===========================================================================
# Required Routing Contract Regression Tests (6 tests)
# ===========================================================================

def test_exact_live_visualization_query_routes_to_data():
    """Test 1 — Exact live regression: the exact multi-chart Olist query must route to data, not ml."""
    from poc_kanini.graphs.supervisor import SupervisorRouter
    router = SupervisorRouter()
    decision = router._deterministic_route(EXACT_OLIST_MULTI_CHART_QUERY)
    assert decision is not None, "Deterministic route must return a decision (not defer to LLM)"
    assert decision.route == "data", f"Expected 'data', got '{decision.route}'"
    assert decision.confidence == 1.0


def test_simple_bar_chart_request_routes_to_data():
    """Test 2 — Simple visualization: 'Create a bar chart showing order counts by order_status' → data."""
    from poc_kanini.graphs.supervisor import SupervisorRouter
    router = SupervisorRouter()
    decision = router._deterministic_route("Create a bar chart showing order counts by order_status.")
    assert decision is not None
    assert decision.route == "data"


def test_explicit_ml_training_routes_to_ml():
    """Test 3 — ML stays ML: 'Train a classification model using feature1, feature2, and churn.' → ml."""
    from poc_kanini.graphs.supervisor import SupervisorRouter
    router = SupervisorRouter()
    decision = router._deterministic_route("Train a classification model using feature1, feature2, and churn.")
    assert decision is not None
    assert decision.route == "ml"


def test_explicit_ml_prediction_routes_to_ml():
    """Test 4 — ML prediction stays ML: 'Use the model you just trained to predict churn for feature1=8 and feature2=9.' → ml."""
    from poc_kanini.graphs.supervisor import SupervisorRouter
    router = SupervisorRouter()
    decision = router._deterministic_route("Use the model you just trained to predict churn for feature1=8 and feature2=9.")
    assert decision is not None
    assert decision.route == "ml"


def test_dataset_terminology_does_not_override_visualization_intent():
    """Test 5 — Dataset terminology must not override visualization intent: 'Plot the numeric columns in this dataset.' → data."""
    from poc_kanini.graphs.supervisor import SupervisorRouter
    router = SupervisorRouter()
    decision = router._deterministic_route("Plot the numeric columns in this dataset.")
    assert decision is not None
    assert decision.route == "data", f"Expected 'data', got '{decision.route}'"


def test_visualization_request_never_creates_ml_approval():
    """Test 6 — No accidental ML approval: visualization request with CSV data must not create ML approval_required=True."""
    from starlette.testclient import TestClient
    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "Create a bar chart showing order counts by order_status."}],
                "csv_data": SAMPLE_OLIST_CSV,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["approval_required"] is False, "Visualization requests must never trigger ML HITL approval"
        tools_run = [t.get("tool") for t in payload.get("tool_results", [])]
        assert "train_ml_model_tool" not in tools_run, "train_ml_model_tool must not be invoked for visualization requests"
        assert "visualize_dataset_tool" in tools_run, "visualize_dataset_tool must be invoked"


# ===========================================================================
# Column Selection & Multi-Chart Grounding Regression Tests
# ===========================================================================

def test_explicit_order_status_column_selection_not_order_id():
    """'Create a bar chart comparing order counts by order_status' must select order_status as x_field and NOT order_id."""
    service = VisualizationService()
    res = service.visualize(SAMPLE_OLIST_CSV, query="Create a bar chart comparing order counts by order_status.")
    assert len(res) == 1
    v = res[0]
    assert v["chart_type"] == "bar"
    assert v["x_field"] == "order_status", f"Expected x_field='order_status', got '{v.get('x_field')}'"
    assert v["y_field"] == "count"
    # Check aggregation results
    status_map = {d["label"]: d["value"] for d in v["data"]}
    assert status_map.get("delivered") == 4
    assert status_map.get("shipped") == 1
    assert status_map.get("canceled") == 1


def test_dataset_agnostic_syntactic_grouping_resolution():
    """Verify syntactic cues ('by department', 'by category', 'across department') resolve the correct grouping column."""
    service = VisualizationService()

    # HR dataset: "group by department"
    res_hr = service.visualize(DATASET_2_HR, query="group by department")
    assert res_hr[0]["x_field"] == "department"

    # HR dataset: "average salary across department"
    res_hr2 = service.visualize(DATASET_2_HR, query="average salary across department")
    assert res_hr2[0]["x_field"] == "department"
    assert res_hr2[0]["y_field"] == "avg_salary"

    # Transport dataset: "orders by vehicle_type" (even if user says 'orders' on a non-order dataset)
    res_tr = service.visualize(DATASET_5_TRANSPORT, query="trips by vehicle_type")
    assert res_tr[0]["x_field"] == "vehicle_type"
    assert res_tr[0]["y_field"] == "trips"


def test_exact_live_olist_multi_chart_prompt_generates_three_grounded_charts():
    """Exact live Olist multi-chart request must generate 3 distinct charts matching user specifications."""
    service = VisualizationService()
    charts = service.visualize(SAMPLE_OLIST_CSV, query=EXACT_OLIST_MULTI_CHART_QUERY)
    assert len(charts) == 3, f"Expected 3 distinct charts, got {len(charts)}"

    # Chart 1: Bar chart comparing order counts by order_status
    c1 = charts[0]
    assert c1["chart_type"] == "bar"
    assert c1["x_field"] == "order_status"
    c1_map = {d["label"]: d["value"] for d in c1["data"]}
    assert c1_map.get("delivered") == 4

    # Chart 2: Line chart showing order volume over time using order_purchase_timestamp
    c2 = charts[1]
    assert c2["chart_type"] == "line"
    assert c2["x_field"] == "order_purchase_timestamp"
    assert len(c2["data"]) == 3  # 3 distinct months: 2023-01, 2023-02, 2023-03

    # Chart 3: Scatter chart showing relationship between two numeric variables (payment_value, freight_value)
    c3 = charts[2]
    assert c3["chart_type"] == "scatter"
    assert {c3["x_field"], c3["y_field"]} == {"payment_value", "freight_value"}
    assert len(c3["data"]) == 6


def test_e2e_chat_endpoint_honors_order_status_in_payload():
    """End-to-end /api/chat invocation produces visualization with x_field='order_status' and zero ML approval."""
    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "Create a bar chart comparing order counts by order_status."}],
                "csv_data": SAMPLE_OLIST_CSV,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["approval_required"] is False
        assert len(payload.get("visualizations", [])) == 1
        viz = payload["visualizations"][0]
        assert viz["chart_type"] == "bar"
        assert viz["x_field"] == "order_status"
        status_map = {d["label"]: d["value"] for d in viz["data"]}
        assert status_map.get("delivered") == 4

