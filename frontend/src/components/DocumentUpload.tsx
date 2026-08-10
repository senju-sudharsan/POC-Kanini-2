import { useState } from "react";
import { FileUp, Loader2 } from "lucide-react";

type IndexingResult = { document: { metadata: { document_id: string; filename: string; page_count: number }; document_type: string; processing_notes: string[] }; chunk_count: number };

export function DocumentUpload({ onIndexed }: { onIndexed: (documentId: string) => void }) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const upload = async (file: File) => {
    setIsProcessing(true); setStatus(null);
    const form = new FormData(); form.append("file", file);
    try {
      const response = await fetch("/api/documents/index", { method: "POST", body: form });
      const result = await response.json() as IndexingResult | { detail?: string };
      if (!response.ok) throw new Error("detail" in result ? result.detail : "Document indexing failed.");
      if (!("document" in result)) throw new Error("Document indexing returned an invalid result.");
      onIndexed(result.document.metadata.document_id);
      const notes = result.document.processing_notes.length ? ` ${result.document.processing_notes.join(" ")}` : "";
      setStatus(`${result.document.metadata.filename}: ${result.chunk_count} chunks indexed from ${result.document.metadata.page_count} page(s), classified as ${result.document.document_type}.${notes}`);
    } catch (error) { setStatus(error instanceof Error ? error.message : "Document indexing failed."); }
    finally { setIsProcessing(false); }
  };
  return <div className="mt-2 w-full text-left"><label className="flex cursor-pointer items-center justify-center gap-2 rounded-lg border border-dashed border-neutral-600 p-3 text-sm text-neutral-300 hover:border-neutral-400"><FileUp size={16} />{isProcessing ? "Indexing document…" : "Upload a PDF for document Q&A"}<input className="sr-only" type="file" accept="application/pdf,.pdf" disabled={isProcessing} onChange={(event) => { const file = event.target.files?.[0]; if (file) void upload(file); event.currentTarget.value = ""; }} /></label>{isProcessing && <div className="mt-2 flex items-center justify-center gap-2 text-xs text-neutral-400"><Loader2 className="size-3 animate-spin" /> Extracting, embedding, and indexing…</div>}{status && <p className="mt-2 text-center text-xs text-neutral-400" role="status">{status}</p>}<p className="mt-2 text-center text-xs text-neutral-500">Indexed PDFs are used as grounded evidence for questions in this chat.</p></div>;
}
