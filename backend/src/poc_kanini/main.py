import logging
import logging.config
import pathlib
import uuid
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from poc_kanini.core.config import get_settings
from poc_kanini.documents.processor import DocumentProcessor, DocumentValidationError
from poc_kanini.graphs.chat import chat_graph, hybrid_chat_graph
from poc_kanini.models.actions import ActionRequest, ActionResult, ReportPayload
from poc_kanini.models.chat import ApprovalDecisionRequest, ChatMessage, ChatRequest, ChatResponse
from poc_kanini.models.documents import ProcessedDocument
from poc_kanini.multimodal.models import MultimodalAnalysis
from poc_kanini.multimodal.service import MultimodalService
from poc_kanini.multimodal.validator import ImageValidationError
from poc_kanini.ml.models import DatasetProfile, PredictRequest, PredictResponse, TrainRequest, TrainResponse
from poc_kanini.ml.service import MlService
from poc_kanini.rag.service import RagService
from poc_kanini.rag.vector_store import ChromaVectorStore
from poc_kanini.services.report_service import execute_action, generate_report

# ---------------------------------------------------------------------------
# Logging setup — configure once on import, before first log statement
# ---------------------------------------------------------------------------

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            }
        },
        "root": {"level": "INFO", "handlers": ["console"]},
        # Suppress noisy third-party library loggers
        "loggers": {
            "uvicorn.access": {"level": "WARNING"},
            "chromadb": {"level": "WARNING"},
            "httpx": {"level": "WARNING"},
            "httpcore": {"level": "WARNING"},
        },
    }
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

settings = get_settings()
app = FastAPI(title=settings.app_name, version="9.0.0")

# CORS — restrictive by default; override CORS_ORIGINS in production env
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)

ml_service_instance = MlService()
multimodal_service_instance = MultimodalService(settings)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ReportGenerateRequest(BaseModel):
    """Typed request body for /api/reports/generate."""

    report_type: str = "executive_summary"
    user_query: str = ""
    tool_results: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Helper: classify provider errors to appropriate HTTP status codes
# ---------------------------------------------------------------------------

def _provider_http_status(error: Exception) -> int:
    """Map a provider/upstream error to an appropriate HTTP status code."""
    msg = str(error).lower()
    if "api_key" in msg or "api key" in msg or "unauthenticated" in msg or "permission" in msg:
        return 401
    if "quota" in msg or "rate" in msg or "429" in msg:
        return 429
    if "not found" in msg or "404" in msg:
        return 404
    if "timeout" in msg or "deadline" in msg:
        return 504
    return 503


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health() -> JSONResponse:
    """Return a lightweight readiness response without exposing secrets."""
    return JSONResponse(
        {
            "status": "ok",
            "environment": settings.environment,
            "gemini_configured": bool(settings.gemini_api_key),
        }
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Run stateful conversation through the AURA LangGraph hybrid agent."""

    thread_id = request.thread_id or f"thread_{uuid.uuid4().hex[:8]}"
    turn_id = f"turn_{uuid.uuid4().hex[:8]}"
    messages = [
        HumanMessage(content=item.content) if item.role == "user" else AIMessage(content=item.content)
        for item in request.messages
    ]
    attachments = [att.model_dump() for att in request.attachments]

    input_state: dict[str, Any] = {
        "messages": messages,
        "attachments": attachments,
        "step_count": 0,
        "max_steps": 5,
        "thread_id": thread_id,
        "turn_id": turn_id,
        "approval_status": request.approval,
    }

    if "document_id" in request.model_fields_set or "document_ids" in request.model_fields_set:
        doc_ids = list(request.document_ids)
        if request.document_id and request.document_id not in doc_ids:
            doc_ids.insert(0, request.document_id)
        input_state["document_ids"] = doc_ids
    elif not request.document_ids and not request.document_id:
        # Explicit empty document association for this turn
        input_state["document_ids"] = []

    if request.csv_data is not None:
        input_state["csv_data"] = request.csv_data

    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = await hybrid_chat_graph.ainvoke(input_state, config=config)
    except RuntimeError as error:
        logger.error("Chat runtime error for thread %s: %s", thread_id, error)
        status = _provider_http_status(error)
        raise HTTPException(status_code=status, detail="The AI provider could not complete the request.") from error
    except Exception as error:
        logger.error("Unexpected chat error for thread %s: %s", thread_id, error)
        raise HTTPException(
            status_code=502,
            detail="The assistant encountered an unexpected error. Please try again.",
        ) from error

    final_content = str(result["messages"][-1].content) if result.get("messages") else "No response generated."
    activities = result.get("activities") or []
    appr_required = bool(result.get("approval_required", False))
    appr_id = result.get("approval_id")
    appr_reason = result.get("approval_reason")
    operation = result.get("operation") or (result.get("route") if appr_required else None)

    from poc_kanini.graphs.turn_context import get_current_turn_tools

    current_tools = get_current_turn_tools(result)

    return ChatResponse(
        message=ChatMessage(role="assistant", content=final_content),
        thread_id=thread_id,
        approval_required=appr_required,
        approval_id=appr_id,
        approval_reason=appr_reason,
        operation=operation,
        activities=activities,
        citations=result.get("citations") or [],
        tool_results=current_tools,
        warnings=result.get("warnings") or [],
        synthesis_status=result.get("synthesis_status") or "success",
        reports=result.get("reports") or [],
        actions=result.get("actions") or [],
    )


@app.post("/api/chat/approval", response_model=ChatResponse)
async def chat_approval(request: ApprovalDecisionRequest) -> ChatResponse:
    """Submit an explicit human approval or rejection decision for a pending controlled operation."""

    default_msg = (
        "Proceed with approved operation."
        if request.decision == "approved"
        else "Cancel rejected operation."
    )
    chat_request = ChatRequest(
        thread_id=request.thread_id,
        approval=request.decision,
        messages=[ChatMessage(role="user", content=request.message or default_msg)],
    )
    return await chat(chat_request)


@app.post("/api/reports/generate", response_model=ReportPayload)
async def generate_report_endpoint(request: ReportGenerateRequest) -> ReportPayload:
    """Generate a structured domain report payload."""
    valid_types = {"executive_summary", "dataset_analysis", "document_analysis", "image_analysis"}
    if request.report_type not in valid_types:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid report_type '{request.report_type}'. Valid types: {sorted(valid_types)}",
        )
    return generate_report(
        report_type=request.report_type,
        tool_results=request.tool_results,
        user_query=request.user_query,
        citations=request.citations,
    )


@app.post("/api/actions/execute", response_model=ActionResult)
async def execute_action_endpoint(request: ActionRequest) -> ActionResult:
    """Execute a safe controlled demonstration action."""
    return execute_action(request)


@app.post("/api/documents/process", response_model=ProcessedDocument)
async def process_document(file: UploadFile = File(...)) -> ProcessedDocument:
    """Process one PDF into structured, provenance-preserving pages."""

    content = await file.read()
    safe_filename = pathlib.Path(file.filename or "upload.pdf").name
    try:
        return DocumentProcessor().process(content, safe_filename, file.content_type)
    except DocumentValidationError as error:
        status = 413 if "exceeds" in str(error) else 400
        raise HTTPException(status_code=status, detail=str(error)) from error


def rag_service() -> RagService:
    """Build the persistent local RAG service only when document RAG is used."""
    return RagService(settings, ChromaVectorStore(settings.rag_vector_store_dir))


@app.post("/api/documents/index")
async def index_document(file: UploadFile = File(...)) -> dict[str, object]:
    """Process and index a PDF into the vector store."""

    content = await file.read()
    safe_filename = pathlib.Path(file.filename or "upload.pdf").name
    try:
        document = DocumentProcessor().process(content, safe_filename, file.content_type)
        chunk_count = await rag_service().index_document(document)
        return {"document": document, "chunk_count": chunk_count}
    except DocumentValidationError as error:
        status = 413 if "exceeds" in str(error) else 400
        raise HTTPException(status_code=status, detail=str(error)) from error
    except RuntimeError as error:
        logger.error("RAG indexing runtime error: %s", error)
        raise HTTPException(status_code=503, detail="Document indexing is unavailable. Check the embedding service configuration.") from error
    except Exception as error:
        logger.error("RAG indexing provider failure; status=%s", _provider_http_status(error))
        raise HTTPException(
            status_code=_provider_http_status(error),
            detail="Document indexing is unavailable. Check the embedding service configuration.",
        ) from error


@app.post("/api/documents/chat")
async def document_chat(request: dict[str, str]) -> dict[str, object]:
    """Answer a question using retrieved PDF evidence and return real citations."""

    question = request.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=422, detail="question is required.")
    try:
        return (await rag_service().answer(question, request.get("document_id"))).model_dump()
    except RuntimeError as error:
        logger.error("Document chat error: %s", error)
        raise HTTPException(status_code=503, detail="Document retrieval is unavailable.") from error


@app.post("/api/ml/profile", response_model=DatasetProfile)
async def ml_profile(request: list[dict[str, Any]]) -> DatasetProfile:
    """Extract structural details and statistics from a tabular dataset."""

    try:
        return ml_service_instance.profile(request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.error("ML profiling error: %s", error)
        raise HTTPException(status_code=500, detail="An error occurred during profiling.") from error


@app.post("/api/ml/train", response_model=TrainResponse)
async def ml_train(request: TrainRequest) -> TrainResponse:
    """Train a baseline classifier or regressor model on a custom dataset."""

    try:
        return ml_service_instance.train(
            data=request.data,
            target=request.target,
            task=request.task,
            model_type=request.model_type,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.error("ML training error: %s", error)
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/api/ml/predict", response_model=PredictResponse)
async def ml_predict(request: PredictRequest) -> PredictResponse:
    """Generate model predictions using a cached process-lifetime estimator."""

    try:
        return ml_service_instance.predict(
            model_id=request.model_id,
            data=request.data,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.error("ML prediction error for model %s: %s", request.model_id, error)
        raise HTTPException(status_code=500, detail="An error occurred during prediction.") from error


@app.post("/api/multimodal/analyze", response_model=MultimodalAnalysis)
async def multimodal_analyze(
    file: UploadFile = File(...),
    question: str = "Describe what you see in this image.",
) -> MultimodalAnalysis:
    """Analyse an uploaded image using Gemini multimodal understanding."""

    content = await file.read()
    safe_filename = pathlib.Path(file.filename or "upload").name
    mime_type = file.content_type or ""
    try:
        return await multimodal_service_instance.analyze(
            content=content,
            mime_type=mime_type,
            question=question,
            filename=safe_filename,
        )
    except ImageValidationError as error:
        status = 413 if "exceeds" in str(error) else 400
        raise HTTPException(status_code=status, detail=str(error)) from error
    except RuntimeError as error:
        logger.error("Multimodal analysis error: %s", error)
        status = _provider_http_status(error)
        raise HTTPException(status_code=status, detail="Image analysis is unavailable.") from error
    except Exception as error:
        logger.error("Unexpected multimodal error: %s", error)
        raise HTTPException(status_code=500, detail="An error occurred during image analysis.") from error


# ---------------------------------------------------------------------------
# Static frontend (production mode only — not served in dev)
# ---------------------------------------------------------------------------

def _frontend_build_path() -> pathlib.Path:
    """Resolve the frontend build relative to the backend source layout."""
    return pathlib.Path(__file__).resolve().parents[2] / settings.frontend_build_dir


build_path = _frontend_build_path()
if build_path.is_dir() and (build_path / "index.html").is_file():
    logger.info("Serving frontend build from: %s", build_path)
    app.mount("/app", StaticFiles(directory=build_path, html=True), name="frontend")
