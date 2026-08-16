"""Phase 5 tool registry and exports.

Provides clean, independently testable tool interfaces for RAG, dataset profiling,
ML training & prediction, and multimodal image analysis.
These tools are designed for future consumption by Phase 6 LangGraph agent routing.
"""

from poc_kanini.tools.data_tools import profile_dataset_tool
from poc_kanini.tools.visualization_tools import visualize_dataset_tool
from poc_kanini.tools.ml_tools import predict_ml_model_tool, train_ml_model_tool
from poc_kanini.tools.multimodal_tools import analyze_image_tool
from poc_kanini.tools.rag_tools import search_document_evidence

ALL_TOOLS = [
    search_document_evidence,
    profile_dataset_tool,
    visualize_dataset_tool,
    train_ml_model_tool,
    predict_ml_model_tool,
    analyze_image_tool,
]

TOOLS_BY_NAME = {tool.name: tool for tool in ALL_TOOLS}

__all__ = [
    "ALL_TOOLS",
    "TOOLS_BY_NAME",
    "search_document_evidence",
    "profile_dataset_tool",
    "visualize_dataset_tool",
    "train_ml_model_tool",
    "predict_ml_model_tool",
    "analyze_image_tool",
]
