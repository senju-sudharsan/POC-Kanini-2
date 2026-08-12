import { useState } from "react";
import { FileUp, Loader2, CheckCircle2 } from "lucide-react";

type IndexingResult = {
  document: {
    metadata: { document_id: string; filename: string; page_count: number };
    document_type: string;
    processing_notes: string[];
  };
  chunk_count: number;
};

export function DocumentUpload({ onIndexed }: { onIndexed: (documentId: string, filename?: string) => void }) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [indexedDocName, setIndexedDocName] = useState<string | null>(null);

  const upload = async (file: File) => {
    setIsProcessing(true);
    setStatus(null);
    const form = new FormData();
    form.append("file", file);

    try {
      const response = await fetch("/api/documents/index", { method: "POST", body: form });
      const result = (await response.json()) as IndexingResult | { detail?: string };
      if (!response.ok) throw new Error("detail" in result ? result.detail : "Document indexing failed.");
      if (!("document" in result)) throw new Error("Document indexing returned an invalid result.");

      const docId = result.document.metadata.document_id;
      const filename = result.document.metadata.filename;
      onIndexed(docId, filename);
      setIndexedDocName(filename);

      const notes = result.document.processing_notes.length ? ` ${result.document.processing_notes.join(" ")}` : "";
      setStatus(
        `Indexed ${filename}: ${result.chunk_count} chunk(s) from ${result.document.metadata.page_count} page(s) [classified: ${result.document.document_type}].${notes}`
      );
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Document indexing failed.");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="mt-2 w-full text-left">
      <label className="flex cursor-pointer items-center justify-center gap-2 rounded-lg border border-dashed border-neutral-600 p-3 text-sm text-neutral-300 hover:border-neutral-400 hover:bg-neutral-700/30 transition">
        <FileUp size={16} className="text-blue-400" />
        {isProcessing ? (
          "Indexing document..."
        ) : indexedDocName ? (
          <span className="flex items-center gap-1.5 text-emerald-400">
            <CheckCircle2 size={16} /> Attached: <strong>{indexedDocName}</strong> (Click to replace)
          </span>
        ) : (
          "Upload a PDF for grounded evidence Q&A"
        )}
        <input
          className="sr-only"
          type="file"
          accept="application/pdf,.pdf"
          disabled={isProcessing}
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void upload(file);
            event.currentTarget.value = "";
          }}
        />
      </label>

      {isProcessing && (
        <div className="mt-2 flex items-center justify-center gap-2 text-xs text-neutral-400">
          <Loader2 className="size-3 animate-spin text-blue-400" /> Extracting pages, generating embeddings, and indexing...
        </div>
      )}

      {status && (
        <p className="mt-2 text-center text-xs text-neutral-300" role="status">
          {status}
        </p>
      )}
      <p className="mt-1 text-center text-[11px] text-neutral-500">
        Uploaded documents are automatically linked to your thread for grounded answer generation.
      </p>
    </div>
  );
}
