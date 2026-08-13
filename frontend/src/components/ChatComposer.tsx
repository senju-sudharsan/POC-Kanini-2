import { useCallback, useRef, useState } from "react";
import { ArrowUp, FileText, Image as ImageIcon, Paperclip, Square, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { ImageAttachment } from "@/lib/chat";

export function ChatComposer({
  onSubmit,
  onCancel,
  isLoading,
  onFocusChange,
  onTypingChange,
  onIndexed,
  onUnlinkDocument,
  documentName,
  auraState,
}: {
  onSubmit: (value: string, attachments?: ImageAttachment[]) => void;
  onCancel?: () => void;
  isLoading: boolean;
  onFocusChange?: (focused: boolean) => void;
  onTypingChange?: (typing: boolean) => void;
  onIndexed?: (documentId: string, filename?: string) => void;
  onUnlinkDocument?: () => void;
  documentName?: string | null;
  auraState?: string;
}) {
  const [value, setValue] = useState("");
  const [attachments, setAttachments] = useState<ImageAttachment[]>([]);
  const [isUploadingDoc, setIsUploadingDoc] = useState(false);
  const [documentUploadError, setDocumentUploadError] = useState<string | null>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const docInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleTextChange = (text: string) => {
    setValue(text);
    onTypingChange?.(Boolean(text.trim()));
  };

  const handleImageFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || !files.length) return;

    const newAttachments: ImageAttachment[] = [];
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) continue;

      const base64Data = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          const res = reader.result as string;
          resolve(res.split(",")[1] || "");
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });

      newAttachments.push({
        filename: file.name,
        mime_type: file.type,
        data: base64Data,
      });
    }

    setAttachments((prev) => [...prev, ...newAttachments]);
    if (e.target) e.target.value = "";
  };

  const handleDocFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || !files.length) return;
    const file = files[0];
    if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
      setDocumentUploadError("Only PDF files can be attached.");
      e.target.value = "";
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setDocumentUploadError("The PDF exceeds the 20 MiB upload limit.");
      e.target.value = "";
      return;
    }

    setIsUploadingDoc(true);
    setDocumentUploadError(null);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/documents/index", {
        method: "POST",
        body: formData,
      });
      const data = await res.json().catch(() => null) as { document?: { metadata?: { document_id?: string; filename?: string } }; detail?: string } | null;
      if (!res.ok) throw new Error(data?.detail || "Document indexing failed. Please try again.");
      const docId = data?.document?.metadata?.document_id;
      const filename = data?.document?.metadata?.filename || file.name;
      if (!docId) throw new Error("Document indexing did not return a document ID.");
      onIndexed?.(docId, filename);
    } catch (err) {
      setDocumentUploadError(err instanceof Error ? err.message : "Document indexing failed. Please try again.");
    } finally {
      setIsUploadingDoc(false);
      if (e.target) e.target.value = "";
    }
  };

  const handleSubmit = useCallback(() => {
    const trimmed = value.trim();
    if ((!trimmed && !attachments.length) || isLoading) return;

    onSubmit(trimmed, attachments.length ? attachments : undefined);
    setValue("");
    setAttachments([]);
    onTypingChange?.(false);

    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }, [value, attachments, isLoading, onSubmit, onTypingChange]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="composer-dock">
      <div className="composer-wrap" data-state={auraState || "idle"}>
        {/* Hidden inputs */}
        <input
          type="file"
          ref={imageInputRef}
          className="hidden"
          accept="image/jpeg,image/png,image/webp"
          multiple
          onChange={handleImageFileSelect}
        />
        <input
          type="file"
          ref={docInputRef}
          className="hidden"
          accept="application/pdf"
          onChange={handleDocFileSelect}
        />

        {/* Attachment preview chips */}
        {(attachments.length > 0 || documentName || isUploadingDoc || documentUploadError) && (
          <div className="px-4 pt-2.5 flex flex-wrap gap-2 items-center">
            {documentName && (
              <div className="flex items-center gap-1.5 bg-neutral-900 border border-blue-900/50 text-blue-300 text-xs pl-2.5 pr-1.5 py-1 rounded-full">
                <FileText className="w-3.5 h-3.5" />
                <span className="truncate max-w-[140px]">{documentName}</span>
                {onUnlinkDocument && (
                  <button
                    type="button"
                    onClick={onUnlinkDocument}
                    className="hover:text-red-400 ml-0.5"
                    title="Remove document association"
                  >
                    <X className="w-3 h-3" />
                  </button>
                )}
              </div>
            )}
            {isUploadingDoc && (
              <div className="flex items-center gap-1.5 bg-neutral-900 border border-amber-900/50 text-amber-300 text-xs px-2.5 py-1 rounded-full animate-pulse">
                <span>Indexing PDF...</span>
              </div>
            )}
            {documentUploadError && (
              <div className="flex items-center gap-1.5 bg-red-950/40 border border-red-800/60 text-red-200 text-xs px-2.5 py-1 rounded-full" role="alert">
                <span>{documentUploadError}</span>
                <button type="button" onClick={() => setDocumentUploadError(null)} className="hover:text-white" aria-label="Dismiss document upload error"><X className="w-3 h-3" /></button>
              </div>
            )}
            {attachments.map((att, idx) => (
              <div
                key={idx}
                className="flex items-center gap-1.5 bg-neutral-900 border border-neutral-700 text-neutral-300 text-xs px-2.5 py-1 rounded-full"
              >
                <ImageIcon className="w-3.5 h-3.5 text-indigo-400" />
                <span className="truncate max-w-[120px]">{att.filename}</span>
                <button
                  type="button"
                  onClick={() => setAttachments((prev) => prev.filter((_, i) => i !== idx))}
                  className="hover:text-red-400 ml-0.5"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="composer">
          <textarea
            ref={textareaRef}
            value={value}
            rows={1}
            onChange={(e) => handleTextChange(e.target.value)}
            onFocus={() => onFocusChange?.(true)}
            onBlur={() => onFocusChange?.(false)}
            onKeyDown={handleKeyDown}
            placeholder="Ask AURA anything..."
            aria-label="Message AURA"
          />
          <div className="composer-tools">
            <div className="flex items-center gap-1">
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="h-8 w-8 text-neutral-400 hover:text-neutral-200"
                onClick={() => docInputRef.current?.click()}
                disabled={isUploadingDoc}
                title="Attach PDF Document"
                aria-label="Attach PDF Document"
              >
                <FileText className="w-4 h-4" />
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="h-8 w-8 text-neutral-400 hover:text-neutral-200"
                onClick={() => imageInputRef.current?.click()}
                title="Attach Image"
                aria-label="Attach Image"
              >
                <Paperclip className="w-4 h-4" />
              </Button>
            </div>

            {isLoading ? (
              <Button
                type="button"
                size="icon"
                className="h-9 w-9 rounded-xl bg-neutral-700 text-neutral-200 hover:bg-neutral-600"
                onClick={onCancel}
                title="Stop Response"
                aria-label="Stop Response"
              >
                <Square className="w-4 h-4 fill-current" />
              </Button>
            ) : (
              <Button
                type="button"
                size="icon"
                className="send-button h-9 w-9"
                disabled={!value.trim() && !attachments.length}
                onClick={handleSubmit}
                aria-label="Send message"
              >
                <ArrowUp className="w-4 h-4" />
              </Button>
            )}
          </div>
        </div>
      </div>
      <p>AURA — Agentic Understanding & Retrieval Assistant. Verify important decisions.</p>
    </div>
  );
}
