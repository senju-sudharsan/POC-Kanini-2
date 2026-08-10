import pathlib

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage

from poc_kanini.core.config import get_settings
from poc_kanini.documents.processor import DocumentProcessor, DocumentValidationError
from poc_kanini.graphs.chat import chat_graph
from poc_kanini.models.chat import ChatRequest, ChatResponse
from poc_kanini.models.documents import ProcessedDocument
from poc_kanini.rag.service import RagService
from poc_kanini.rag.vector_store import ChromaVectorStore

settings = get_settings()
app = FastAPI(title=settings.app_name)


@app.get("/api/health")
async def health() -> JSONResponse:
    """Return a lightweight readiness response without exposing secrets."""

    return JSONResponse({"status": "ok", "environment": settings.environment})


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Run the Phase 1 text conversation through the Gemini LangGraph workflow."""

    messages = [
        HumanMessage(content=item.content) if item.role == "user" else AIMessage(content=item.content)
        for item in request.messages
    ]
    try:
        result = await chat_graph.ainvoke({"messages": messages})
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail="Gemini could not complete the request.") from error

    return ChatResponse(message={"role": "assistant", "content": str(result["messages"][-1].content)})


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


def frontend_build_path() -> pathlib.Path:
    """Resolve the frontend build relative to the backend source layout."""

    return pathlib.Path(__file__).resolve().parents[2] / settings.frontend_build_dir


build_path = frontend_build_path()
if build_path.is_dir() and (build_path / "index.html").is_file():
    app.mount("/app", StaticFiles(directory=build_path, html=True), name="frontend")
