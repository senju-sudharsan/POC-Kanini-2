import { Check, ShieldCheck, X } from "lucide-react";
import { Button } from "@/components/ui/button";

export function ApprovalCard({
  reason,
  decision,
  onDecision,
}: {
  reason: string;
  decision?: "approved" | "rejected" | null;
  onDecision: (value: "approved" | "rejected") => void;
}) {
  if (decision) {
    return (
      <div className="approval-resolved" role="status">
        {decision === "approved" ? (
          <>
            <Check className="w-4 h-4 text-emerald-400" />
            <span className="text-emerald-400 font-medium">Operation Approved & Executed</span>
          </>
        ) : (
          <>
            <X className="w-4 h-4 text-amber-400" />
            <span className="text-amber-400 font-medium">Operation Declined</span>
          </>
        )}
      </div>
    );
  }

  return (
    <section className="approval my-4" aria-labelledby="approval-title">
      <div className="approval-icon">
        <ShieldCheck className="w-5 h-5 text-indigo-400" />
      </div>
      <div className="approval-copy">
        <span className="eyebrow">Human Approval Required</span>
        <h2 id="approval-title" className="text-sm font-medium text-neutral-100 mt-1">
          {reason}
        </h2>
        <p className="text-xs text-neutral-400 mt-1">
          AURA requires explicit authorization before executing controlled operations.
        </p>
      </div>
      <div className="approval-actions">
        <Button
          variant="ghost"
          className="text-xs text-neutral-400 hover:text-neutral-200 h-8 px-3"
          onClick={() => onDecision("rejected")}
        >
          Decline
        </Button>
        <Button
          className="bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs h-8 px-4"
          onClick={() => onDecision("approved")}
        >
          Approve Operation
        </Button>
      </div>
    </section>
  );
}
