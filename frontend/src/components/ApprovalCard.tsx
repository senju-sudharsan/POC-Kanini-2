import { Check, Loader2, ShieldAlert, ShieldCheck, X } from "lucide-react";
import { Button } from "@/components/ui/button";

export interface ApprovalCardProps {
  reason: string;
  approvalId?: string | null;
  operation?: string | null;
  decision?: "approved" | "rejected" | null;
  isLoading?: boolean;
  onDecision?: (value: "approved" | "rejected") => void;
}

export function ApprovalCard({
  reason,
  approvalId,
  operation = "ml",
  decision,
  isLoading = false,
  onDecision,
}: ApprovalCardProps) {
  const displayOperation = operation === "ml" ? "Machine Learning (ML)" : operation || "Controlled Operation";

  if (decision) {
    return (
      <div
        className={`approval-resolved-card ${
          decision === "approved"
            ? "border-emerald-500/30 bg-emerald-950/20 shadow-emerald-500/5"
            : "border-purple-500/30 bg-purple-950/20 shadow-purple-500/5"
        } p-3 rounded-xl border flex items-center justify-between gap-3 text-xs my-3 shadow-md animate-in fade-in duration-300`}
      >
        <div className="flex items-center gap-2.5">
          {decision === "approved" ? (
            <div className="w-6 h-6 rounded-full bg-emerald-500/20 flex items-center justify-center text-emerald-400">
              <Check className="w-3.5 h-3.5" />
            </div>
          ) : (
            <div className="w-6 h-6 rounded-full bg-purple-500/20 flex items-center justify-center text-purple-400">
              <X className="w-3.5 h-3.5" />
            </div>
          )}
          <div>
            <div className={`font-medium ${decision === "approved" ? "text-emerald-300" : "text-purple-300"}`}>
              {decision === "approved" ? "Operation Approved & Executed" : "Operation Declined"}
            </div>
            <div className="text-[11px] text-neutral-400 mt-0.5">
              {displayOperation} {approvalId ? `• ${approvalId}` : ""}
            </div>
          </div>
        </div>
        <span
          className={`px-2 py-0.5 text-[10px] rounded-md uppercase font-mono font-semibold tracking-wider ${
            decision === "approved"
              ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
              : "bg-purple-500/20 text-purple-300 border border-purple-500/30"
          }`}
        >
          {decision}
        </span>
      </div>
    );
  }

  return (
    <section
      className="approval-card my-4 p-4 rounded-xl border border-[#717cff]/30 bg-gradient-to-b from-[#141522] via-[#11131a] to-[#0d0f14] shadow-xl shadow-black/50 text-neutral-200"
      aria-labelledby="approval-title"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-indigo-500/15 border border-indigo-500/30 flex items-center justify-center text-indigo-400 shrink-0 shadow-sm shadow-indigo-500/20">
            <ShieldAlert className="w-4 h-4 text-[#865cff]" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-mono uppercase tracking-widest text-[#a88cff] font-semibold flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-[#4d8dff] shadow-[0_0_6px_#4d8dff] animate-pulse" />
                Human Approval Required
              </span>
            </div>
            <h3 id="approval-title" className="text-sm font-medium text-neutral-100 mt-0.5">
              Controlled Operation: <span className="text-[#865cff] font-semibold uppercase">{operation || "ml"}</span>
            </h3>
          </div>
        </div>

        {approvalId && (
          <span className="text-[10px] font-mono text-neutral-400 bg-neutral-900/80 border border-neutral-800 px-2 py-0.5 rounded-md shrink-0">
            {approvalId}
          </span>
        )}
      </div>

      <div className="mt-3 p-3 rounded-lg bg-[#0b0c10]/80 border border-[#232634] text-xs text-neutral-300">
        <p className="font-medium text-neutral-200">{reason}</p>
        <p className="text-[11px] text-neutral-400 mt-1.5">
          AURA requires explicit authorization before training models or executing controlled actions.
        </p>
      </div>

      <div className="approval-actions flex items-center justify-end gap-2.5 mt-3.5 pt-2 border-t border-[#1f222e]">
        <Button
          variant="ghost"
          disabled={isLoading}
          className="text-xs text-neutral-300 hover:text-white hover:bg-neutral-800/80 border border-neutral-700/80 h-8 px-3.5 transition-colors cursor-pointer"
          onClick={() => onDecision?.("rejected")}
        >
          <X className="w-3.5 h-3.5 mr-1 text-neutral-400" />
          Decline
        </Button>
        <Button
          disabled={isLoading}
          className="bg-gradient-to-r from-[#4d8dff] to-[#865cff] hover:from-[#3d7ded] hover:to-[#764cef] text-white font-medium text-xs h-8 px-4 shadow-md shadow-indigo-500/25 transition-all cursor-pointer flex items-center gap-1.5 border-0"
          onClick={() => onDecision?.("approved")}
        >
          {isLoading ? (
            <>
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              Submitting...
            </>
          ) : (
            <>
              <ShieldCheck className="w-3.5 h-3.5" />
              Approve Operation
            </>
          )}
        </Button>
      </div>
    </section>
  );
}
