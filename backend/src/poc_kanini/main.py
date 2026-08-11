import pathlib
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage

from poc_kanini.core.config import get_settings
from poc_kanini.documents.processor import DocumentProcessor, DocumentValidationError
from poc_kanini.graphs.chat import chat_graph, hybrid_chat_graph
from poc_kanini.models.chat import ChatMessage, ChatRequest, ChatResponse
from poc_kanini.models.documents import ProcessedDocument
from poc_kanini.rag.service import RagService
from poc_kanini.rag.vector_store import ChromaVectorStore
from poc_kanini.ml.service import MlService
from poc_kanini.ml.models import DatasetProfile, TrainResponse, TrainRequest, PredictRequest, PredictResponse
from poc_kanini.multimodal.service import MultimodalService
from poc_kanini.multimodal.models import MultimodalAnalysis
from poc_kanini.multimodal.validator import ImageValidationError

settings = get_settings()
app = FastAPI(title=settings.app_name)
ml_service_instance = MlService()
multimodal_service_instance = MultimodalService(settings)


@app.get("/api/health")
async def health() -> JSONResponse:
    """Return a lightweight readiness response without exposing secrets."""

    return JSONResponse({"status": "ok", "environment": settings.environment})


import uuid

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Run stateful conversation through Phase 7 LangGraph hybrid agent workflow with checkpointing and HITL."""

    thread_id = request.thread_id or f"thread_{uuid.uuid4().hex[:8]}"
    messages = [
        HumanMessage(content=item.content) if item.role == "user" else AIMessage(content=item.content)
        for item in request.messages
    ]
    attachments = [att.model_dump() for att in request.attachments]

    input_state = {
        "messages": messages,
        "attachments": attachments,
        "step_count": 0,
        "max_steps": 5,
        "thread_id": thread_id,
    }
    if request.approval:
        input_state["approval_status"] = request.approval

    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = await hybrid_chat_graph.ainvoke(input_state, config=config)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"The assistant could not complete the request: {error}") from error

    final_content = str(result["messages"][-1].content) if result.get("messages") else "No response generated."
    activities = result.get("activities") or []
    appr_required = bool(result.get("approval_required", False))
    appr_id = result.get("approval_id")
    appr_reason = result.get("approval_reason")

    return ChatResponse(
        message=ChatMessage(role="assistant", content=final_content),
        thread_id=thread_id,
        approval_required=appr_required,
        approval_id=appr_id,
        approval_reason=appr_reason,
        activities=activities,
    )


@app.post("/api/documents/process", response_model=ProcessedDocument)
async def process_document(file: UploadFile = File(...)) -> ProcessedDocument:
    """Process one PDF into structured, provenance-preserving pages."""

    content = await file.read()
    try:
        return DocumentProcessor().process(content, file.filename or "upload.pdf", file.content_type)
    except DocumentValidationError as error:
        status = 413 if "exceeds" in str(error) else 400
        raise HTTPException(status_code=status, detail=str(error)) from error


def rag_service() -> RagService:
    """Build the persistent local RAG service only when document RAG is used."""

    return RagService(settings, ChromaVectorStore(settings.rag_vector_store_dir))


@app.post("/api/documents/index")
async def index_document(file: UploadFile = File(...)) -> dict[str, object]:
    """Process and index a PDF without changing the Phase 2 process endpoint."""

    content = await file.read()
    try:
        document = DocumentProcessor().process(content, file.filename or "upload.pdf", file.content_type)
        chunk_count = await rag_service().index_document(document)
        return {"document": document, "chunk_count": chunk_count}
    except DocumentValidationError as error:
        status = 413 if "exceeds" in str(error) else 400
        raise HTTPException(status_code=status, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/api/documents/chat")
async def document_chat(request: dict[str, str]) -> dict[str, object]:
    """Answer a question using retrieved PDF evidence and return real citations."""

    question = request.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=422, detail="question is required.")
    try:
        return (await rag_service().answer(question, request.get("document_id"))).model_dump()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/api/ml/profile", response_model=DatasetProfile)
async def ml_profile(request: list[dict[str, Any]]) -> DatasetProfile:
    """Extract structural details and statistics from a tabular dataset."""

    try:
        return ml_service_instance.profile(request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
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
        raise HTTPException(status_code=500, detail="An error occurred during prediction.") from error

@app.post("/api/multimodal/analyze", response_model=MultimodalAnalysis)
async def multimodal_analyze(
    file: UploadFile = File(...),
    question: str = "Describe what you see in this image.",
) -> MultimodalAnalysis:
    """Analyse an uploaded image using Gemini multimodal understanding."""

    content = await file.read()
    filename = file.filename or "upload"
    mime_type = file.content_type or ""
    try:
        return await multimodal_service_instance.analyze(
            content=content,
            mime_type=mime_type,
            question=question,
            filename=filename,
        )
    except ImageValidationError as error:
        status = 413 if "exceeds" in str(error) else 400
        raise HTTPException(status_code=status, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="An error occurred during image analysis.") from error


def frontend_build_path() -> pathlib.Path:
    """Resolve the frontend build relative to the backend source layout."""

    return pathlib.Path(__file__).resolve().parents[2] / settings.frontend_build_dir


build_path = frontend_build_path()
if build_path.is_dir() and (build_path / "index.html").is_file():
    app.mount("/app", StaticFiles(directory=build_path, html=True), name="frontend")
