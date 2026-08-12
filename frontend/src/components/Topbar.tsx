import { FileText, RotateCcw, X } from "lucide-react";
import { Button } from "@/components/ui/button";

export function Topbar({
  documentName,
  onResetChat,
  onUnlinkDocument,
}: {
  documentName?: string | null;
  onResetChat?: () => void;
  onUnlinkDocument?: () => void;
}) {
  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-mark">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="9" />
            <circle cx="12" cy="12" r="4" />
          </svg>
        </div>
        <span>AURA</span>
      </div>

      <div className="flex items-center gap-3">
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
