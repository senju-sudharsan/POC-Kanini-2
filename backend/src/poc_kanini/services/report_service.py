"""Phase 8 Report generation and safe controlled action abstraction service."""

import logging
import uuid
from typing import Any

from poc_kanini.models.actions import ActionRequest, ActionResult, ReportPayload, ReportSection

logger = logging.getLogger(__name__)


def generate_report(
    report_type: str = "executive_summary",
    tool_results: list[dict[str, Any]] | None = None,
    user_query: str = "",
    citations: list[dict[str, Any]] | None = None,
) -> ReportPayload:
    """Synthesize a structured report payload from tool results and user query."""
    tool_results = tool_results or []
    citations = citations or []

    sections: list[ReportSection] = []
    metrics: dict[str, Any] = {}
    recommendations: list[str] = []
    citation_labels: list[str] = [c.get("label", "") for c in citations if c.get("label")]

    # Extract metrics and sections from tool outputs
    for item in tool_results:
        tool_name = item.get("tool")
        res = item.get("result") or {}

        if tool_name == "profile_dataset_tool":
            sections.append(
                ReportSection(
                    title="Dataset Structure & Profiling",
                    content=f"Dataset contains {res.get('row_count', 0)} rows across {res.get('column_count', 0)} columns.",
                    bullet_points=[
                        f"Numeric columns: {', '.join(res.get('numeric_columns', [])) or 'None'}",
                        f"Categorical columns: {', '.join(res.get('categorical_columns', [])) or 'None'}",
                        f"Datetime/Timestamp columns: {', '.join(res.get('datetime_columns', [])) or 'None'}",
                    ],
                )
            )
            metrics["row_count"] = res.get("row_count", 0)
            metrics["column_count"] = res.get("column_count", 0)
            recommendations.append("Inspect missing values and target column distribution before training.")

        elif tool_name == "train_ml_model_tool":
            m = res.get("metrics") or {}
            sections.append(
                ReportSection(
                    title="Machine Learning Evaluation",
                    content=f"Trained {res.get('model_type', 'estimator')} on target '{res.get('target', 'label')}'.",
                    bullet_points=[
                        f"Task: {res.get('task', 'supervised')}",
                        f"Model ID: {res.get('model_id', 'N/A')}",
                        f"Evaluation Metrics: {m}",
                    ],
                )
            )
            metrics.update(m)
            recommendations.append("Deploy trained model for batch prediction or refine hyper-parameters.")

        elif tool_name == "search_document_evidence":
            ev_list = res.get("evidence") or []
            sections.append(
                ReportSection(
                    title="Document Evidence & Provenance",
                    content=res.get("summary", "Retrieved document evidence snippets."),
                    bullet_points=[
                        f"[{e.get('filename')} — Page {e.get('page_number')}]: {e.get('text', '')[:120]}..."
                        for e in ev_list[:3]
                    ],
                )
            )
            recommendations.append(f"Evidence from {len(ev_list)} snippet(s) cited in this report.")

        elif tool_name == "analyze_image_tool":
            if item.get("error"):
                sections.append(
                    ReportSection(
                        title="Multimodal Visual Analysis",
                        content="Visual analysis was not completed due to an error.",
                        bullet_points=[f"Error detail: {item.get('error')}"],
                    )
                )
            else:
                obs = res.get("observations") or []
                sections.append(
                    ReportSection(
                        title="Multimodal Visual Analysis",
                        content=res.get("answer", "Analyzed visual attachment."),
                        bullet_points=obs[:5] if obs else ["Visual inspection completed."],
                    )
                )

    if not sections:
        sections.append(
            ReportSection(
                title="General Analysis",
                content=f"Analysis synthesized for request: {user_query}",
                bullet_points=["Unified assistant processing completed successfully."],
            )
        )
        recommendations.append("Formulate specific questions regarding enterprise documents or datasets.")

    title_map = {
        "dataset_analysis": "Tabular Dataset & ML Analysis Report",
        "document_analysis": "Enterprise Document Evidence & Intelligence Report",
        "image_analysis": "Multimodal Visual Inspection Report",
        "executive_summary": "Unified Enterprise AI Intelligence Summary",
    }
    title = title_map.get(report_type, "Enterprise AI Insight Report")

    return ReportPayload(
        report_type=report_type if report_type in title_map else "executive_summary",
        title=title,
        summary=f"Synthesized report addressing: {user_query or 'Enterprise query'}.",
        sections=sections,
        metrics=metrics,
        citations=citation_labels,
        recommendations=recommendations,
    )


def execute_action(action_request: ActionRequest) -> ActionResult:
    """Execute a safe, local demonstration action within established Phase 8 security boundaries."""
    atype = action_request.action_type
    params = action_request.parameters

    logger.info("Executing safe controlled action: %s", atype)

    if atype == "generate_analysis_summary":
        return ActionResult(
            action_type=atype,
            status="success",
            summary=f"Generated analysis summary for scope '{params.get('scope', 'general')}'.",
            metadata={"summary": f"Analysis summary completed for {params.get('subject', 'request')}."},
        )

    elif atype == "prepare_recommendation":
        rec = f"Recommended next step: Proceed with {params.get('recommendation_type', 'standard workflow')}."
        return ActionResult(
            action_type=atype,
            status="success",
            summary="Actionable recommendation prepared successfully.",
            metadata={"recommendation": rec},
        )

    elif atype == "create_structured_report_payload":
        report = generate_report(
            report_type=params.get("report_type", "executive_summary"),
            user_query=params.get("user_query", ""),
        )
        return ActionResult(
            action_type=atype,
            status="success",
            summary=f"Created structured report payload '{report.title}'.",
            metadata={"report": report.model_dump()},
        )

    elif atype == "profile_dataset":
        return ActionResult(
            action_type=atype,
            status="success",
            summary="Dataset profiling action complete.",
            metadata={"rows": len(params.get("data", [])), "columns": list(params.get("data", [{}])[0].keys()) if params.get("data") else []},
        )

    elif atype == "train_model":
        model_id = str(uuid.uuid4())
        return ActionResult(
            action_type=atype,
            status="success",
            summary=f"Model training action executed (model_id: {model_id}).",
            metadata={"model_id": model_id, "target": params.get("target", "y")},
        )

    return ActionResult(
        action_type=atype,
        status="failed",
        summary=f"Unknown or unsupported action type '{atype}'.",
        metadata={},
    )
