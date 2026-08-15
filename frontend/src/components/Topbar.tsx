import { FileText, RotateCcw, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { getGeminiUsagePresentation } from "@/lib/geminiUsage";
import type { SynthesisStatus } from "@/lib/chat";

export function Topbar({
  documentName,
  onResetChat,
  onUnlinkDocument,
  geminiRequestCount = 0,
  lastSynthesisStatus,
}: {
  documentName?: string | null;
  onResetChat?: () => void;
  onUnlinkDocument?: () => void;
  geminiRequestCount?: number;
  lastSynthesisStatus?: SynthesisStatus;
}) {
  const usage = getGeminiUsagePresentation(lastSynthesisStatus);
  const usageTone = usage.tone === "limited" ? "text-amber-300 border-amber-500/40 bg-amber-950/40" : usage.tone === "warning" ? "text-amber-300 border-amber-500/25 bg-amber-950/25" : "text-blue-300 border-blue-800/40 bg-blue-950/30";
  return (
    <header className="topbar">
      <div className="brand">
        <span>AURA</span>
      </div>

      <div className="flex items-center gap-3">
        <div className={`hidden md:flex items-center gap-2 rounded-full border px-2.5 py-1 text-[10px] ${usageTone}`} title={usage.detail} aria-label={`Gemini usage: ${usage.label}. AURA chat requests this session: ${geminiRequestCount}.`}>
          <span className="h-1.5 w-1.5 rounded-full bg-current" aria-hidden="true" />
          <span>Gemini: {usage.label}</span>
          <span className="text-neutral-400">AURA requests: {geminiRequestCount}</span>
        </div>
        {documentName && (
          <div className="flex items-center gap-1.5 text-xs text-blue-400 bg-blue-950/40 pl-2.5 pr-1.5 py-1 rounded-full border border-blue-800/40">
            <FileText className="w-3.5 h-3.5" />
            <span className="font-medium truncate max-w-[160px]">{documentName}</span>
            {onUnlinkDocument && (
              <button
                type="button"
                onClick={onUnlinkDocument}
                className="hover:text-red-400 p-0.5 rounded-full transition"
                title="Remove document association"
              >
                <X className="w-3.5 h-3.5 text-neutral-400 hover:text-red-400" />
              </button>
            )}
          </div>
        )}
        <div className="header-actions">
          <span className="secure-label hidden sm:flex">
            <span /> Private Session
          </span>
          {onResetChat && (
            <Button
              variant="ghost"
              size="icon"
              className="text-neutral-400 hover:text-neutral-100 hover:bg-neutral-800/60"
              onClick={onResetChat}
              title="Start New Session"
              aria-label="Start new session"
            >
              <RotateCcw className="w-4 h-4" />
            </Button>
          )}
        </div>
      </div>
    </header>
  );
}
