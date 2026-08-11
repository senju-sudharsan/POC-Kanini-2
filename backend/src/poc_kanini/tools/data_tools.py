"""Tabular dataset profiling tool wrapping MlService.profile."""

import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from poc_kanini.ml.service import MlService

logger = logging.getLogger(__name__)


class DatasetProfileInput(BaseModel):
    """Input schema for dataset profiling tool."""

    data: list[dict[str, Any]] | str = Field(
        ...,
        description=(
            "Tabular dataset to profile. Pass either a list of record dicts "
            "(e.g. [{'col1': 1, 'col2': 'A'}, ...]) or raw CSV string content. "
            "Arbitrary filesystem paths are NOT accepted for security."
        ),
    )


@tool("profile_dataset_tool", args_schema=DatasetProfileInput)
def profile_dataset_tool(
    data: list[dict[str, Any]] | str,
) -> dict[str, Any]:
    """Extract structural summary, column types, missing value counts, and statistics from tabular data.

    Use this tool to explore dataset structure before training a machine learning model
    or performing data analysis.
    """
    service = MlService()
    try:
        profile = service.profile(data)
        return profile.model_dump()
    except ValueError as error:
        return {"error": str(error)}
    except Exception as error:
        logger.error("Error in profile_dataset_tool: %s", error)
        return {"error": f"Failed to profile dataset: {error}"}
