"""Tabular dataset visualization tool wrapping VisualizationService."""

import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from poc_kanini.services.visualization_service import VisualizationService

logger = logging.getLogger(__name__)


class DatasetVisualizationInput(BaseModel):
    """Input schema for dataset visualization tool."""

    data: list[dict[str, Any]] | str = Field(
        ...,
        description="Tabular dataset (list of record dicts or CSV string) to analyze and visualize.",
    )
    query: str = Field(
        default="",
        description="User query or visualization goal describing what to plot or analyze.",
    )
    chart_type: str | None = Field(
        default=None,
        description="Optional chart type: 'bar', 'line', 'pie', 'donut', 'scatter', 'table', 'kpi'.",
    )
    x_field: str | None = Field(
        default=None,
        description="Optional explicit column name for X axis or categorical grouping.",
    )
    y_field: str | None = Field(
        default=None,
        description="Optional explicit column name for Y axis or metric calculation.",
    )
    aggregation: str | None = Field(
        default=None,
        description="Optional aggregation method: 'sum', 'mean', 'count'.",
    )
    limit: int | None = Field(
        default=None,
        description="Optional maximum number of data points or categories to return.",
    )


@tool("visualize_dataset_tool", args_schema=DatasetVisualizationInput)
def visualize_dataset_tool(
    data: list[dict[str, Any]] | str,
    query: str = "",
    chart_type: str | None = None,
    x_field: str | None = None,
    y_field: str | None = None,
    aggregation: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Compute and construct dynamic, grounded visualization specifications from tabular data.

    Use this tool whenever the user asks to plot, chart, graph, visualize, show distributions,
    trends over time, relationships between columns, or key metric summaries.
    """
    service = VisualizationService()
    try:
        specs = service.visualize(
            data=data,
            query=query,
            chart_type=chart_type,
            x_field=x_field,
            y_field=y_field,
            aggregation=aggregation,
            limit=limit,
        )
        return {
            "visualizations": specs,
            "count": len(specs),
        }
    except Exception as error:
        logger.error("Error in visualize_dataset_tool: %s", error)
        return {
            "visualizations": [],
            "error": f"Failed to compute visualization: {error}",
        }
