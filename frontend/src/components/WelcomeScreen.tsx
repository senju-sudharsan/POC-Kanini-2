import { useState } from "react";
import { AuraOrb, type AuraState, STATE_LABELS } from "@/components/AuraOrb";
import { FileText, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";

export function WelcomeScreen({
  onIndexed,
  documentName,
  auraState = "idle",
  onPromptSelect,
}: {
  onIndexed?: (documentId: string, filename?: string) => void;
  documentName?: string | null;
  auraState?: AuraState;
  onPromptSelect?: (prompt: string) => void;
}) {
  const [isUploading, setIsUploading] = useState(false);

  const samplePrompts = [
    "What capabilities do you support?",
    "Profile a sample dataset and show columns",
    "Analyze an uploaded image attachment",
    "Search document evidence for policy details",
  ];

  const handleDocUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || !files.length) return;
    const file = files[0];

    setIsUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/documents/index", {
        method: "POST",
        body: formData,
      });
      if (res.ok) {
        const data = await res.json();
        const docId = data.document?.metadata?.document_id;
        const filename = data.document?.metadata?.filename || file.name;
        if (docId) onIndexed?.(docId, filename);
      }
    } catch (err) {
      console.error("Document upload failed:", err);
    } finally {
      setIsUploading(false);
      if (e.target) e.target.value = "";
    }
  };

  return (
    <section className="intro flex flex-col items-center justify-center text-center px-4" aria-labelledby="aura-heading">
      <AuraOrb state={auraState} />
      <div className="intro-copy mt-2">
        <div className="live-state">
          <span /> {STATE_LABELS[auraState] || "Ready"}
        </div>
        <h1 id="aura-heading" className="text-3xl sm:text-4xl font-medium tracking-tight text-neutral-100 mb-2">
          How can I help you today?
        </h1>
        <p className="text-sm text-neutral-400 max-w-md mx-auto">
          Agentic Understanding &amp; Retrieval Assistant for documents, images, datasets, and decision intelligence.
        </p>
      </div>

      {/* Linked Document Pill or Quick Upload */}
      <div className="mt-5">
        <input
          type="file"
          id="welcome-pdf-upload"
          className="hidden"
          accept="application/pdf"
          onChange={handleDocUpload}
        />
        {documentName ? (
          <div className="inline-flex items-center gap-2 bg-[#111318] border border-blue-900/50 text-blue-300 text-xs px-3.5 py-1.5 rounded-full">
            <FileText className="w-4 h-4 text-blue-400" />
            <span>Active Document: <strong>{documentName}</strong></span>
          </div>
        ) : (
          <label htmlFor="welcome-pdf-upload">
            <Button
              type="button"
              variant="ghost"
              className="bg-[#111318] border border-[#282b33] text-neutral-300 hover:text-neutral-100 hover:bg-[#171920] cursor-pointer text-xs rounded-full h-8 px-4 gap-2"
              disabled={isUploading}
              onClick={() => document.getElementById("welcome-pdf-upload")?.click()}
            >
              <FileText className="w-3.5 h-3.5 text-neutral-400" />
              {isUploading ? "Indexing PDF..." : "Link a PDF Document"}
            </Button>
          </label>
        )}
      </div>

      {/* Suggested prompts */}
      {onPromptSelect && (
        <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 gap-2 max-w-xl w-full text-left">
          {samplePrompts.map((prompt, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => onPromptSelect(prompt)}
              className="p-3 rounded-xl border border-[#282b33] bg-[#111318]/80 hover:bg-[#171920] text-xs text-neutral-300 hover:text-neutral-100 transition flex items-center justify-between group"
            >
              <span>{prompt}</span>
              <Sparkles className="w-3.5 h-3.5 text-neutral-500 group-hover:text-indigo-400 shrink-0 ml-2" />
            </button>
          ))}
        </div>
      )}
    </section>
  );
}
