import { FileText, BarChart2, Eye, AlertTriangle } from "lucide-react";
import type { Citation, ToolResult, SynthesisStatus } from "@/lib/chat";

interface Props {
  citations?: Citation[];
  toolResults?: ToolResult[];
  warnings?: string[];
  synthesisStatus?: SynthesisStatus;
}

export function ToolResultCard({ citations, toolResults, warnings, synthesisStatus }: Props) {
  const isDegraded = synthesisStatus === "degraded" || synthesisStatus === "quota_exhausted";
  const statusTitle = synthesisStatus === "quota_exhausted" ? "Gemini usage limit reached" : "Gemini synthesis unavailable";
  const statusDescription = synthesisStatus === "quota_exhausted"
    ? "The AI provider quota is currently exhausted. Wait for the quota to reset or configure another Gemini API key."
    : "AURA is showing a safe fallback based on the available evidence.";
  if (!citations?.length && !toolResults?.length && !warnings?.length && !isDegraded) return null;

  return (
    <div className="mt-4 space-y-3 text-xs">
      {/* Warnings */}
      {isDegraded && (
        <div className="rounded-xl border border-amber-500/40 bg-amber-950/40 p-3 text-amber-200" role="status">
          <div className="flex items-center gap-2 font-medium text-amber-300">
            <AlertTriangle size={14} />
            <span>{statusTitle}</span>
          </div>
          <p className="mt-1 text-amber-200/80">{statusDescription}</p>
        </div>
      )}
      {warnings && warnings.length > 0 && (
        <div className="rounded-xl border border-amber-500/30 bg-amber-950/30 p-3 text-amber-200">
          <div className="flex items-center gap-2 font-medium mb-1.5 text-amber-400">
            <AlertTriangle size={14} />
            <span>Assistant Warnings</span>
          </div>
          <ul className="list-disc list-inside space-y-0.5 text-amber-300/80">
            {warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Citations */}
      {citations && citations.length > 0 && (
        <div className="rounded-xl border border-[#282b33] bg-[#111318] p-3 text-neutral-300">
          <div className="flex items-center gap-2 font-medium mb-2 text-neutral-200">
            <FileText size={14} className="text-blue-400" />
            <span>Document Evidence Citations</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {citations.map((c, i) => (
              <span
                key={i}
                className="inline-flex items-center rounded-md border border-blue-500/30 bg-blue-950/40 px-2.5 py-1 text-[11px] font-medium text-blue-300"
              >
                [{c.label}]
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Tool Results (ML Metrics / Multimodal observations) */}
      {toolResults && toolResults.length > 0 && (
        <div className="space-y-2.5">
          {toolResults.map((tr, idx) => {
            if (tr.tool === "train_ml_model_tool" && tr.result && typeof tr.result === "object") {
              const res = tr.result as Record<string, unknown>;
              const metrics = (res.metrics || {}) as Record<string, number | null>;
              const modelType = (res.model_type || res.model_name || "ML Model") as string;
              const modelId = (res.model_id || "") as string;
              const task = (res.task || "") as string;
              return (
                <div key={idx} className="rounded-xl border border-[#282b33] bg-[#111318] p-3 text-neutral-200">
                  <div className="flex items-center justify-between mb-2 text-indigo-300 font-medium">
                    <span className="flex items-center gap-2">
                      <BarChart2 size={14} />
                      <span>{modelType} ({task})</span>
                    </span>
                    {modelId && <span className="font-mono text-[10px] text-neutral-400">id: {modelId.slice(0, 8)}...</span>}
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 bg-[#171920] p-2.5 rounded-lg border border-[#282b33] text-center">
                    {metrics.accuracy !== undefined && metrics.accuracy !== null && (
                      <div>
                        <div className="text-[10px] text-neutral-400">Accuracy</div>
                        <div className="text-sm font-semibold text-indigo-300">{(metrics.accuracy * 100).toFixed(1)}%</div>
                      </div>
                    )}
                    {metrics.f1 !== undefined && metrics.f1 !== null && (
                      <div>
                        <div className="text-[10px] text-neutral-400">F1 Score</div>
                        <div className="text-sm font-semibold text-indigo-300">{metrics.f1.toFixed(3)}</div>
                      </div>
                    )}
                    {metrics.precision !== undefined && metrics.precision !== null && (
                      <div>
                        <div className="text-[10px] text-neutral-400">Precision</div>
                        <div className="text-sm font-semibold text-indigo-300">{metrics.precision.toFixed(3)}</div>
                      </div>
                    )}
                    {metrics.recall !== undefined && metrics.recall !== null && (
                      <div>
                        <div className="text-[10px] text-neutral-400">Recall</div>
                        <div className="text-sm font-semibold text-indigo-300">{metrics.recall.toFixed(3)}</div>
                      </div>
                    )}
                    {metrics.r2 !== undefined && metrics.r2 !== null && (
                      <div>
                        <div className="text-[10px] text-neutral-400">R² Score</div>
                        <div className="text-sm font-semibold text-indigo-300">{metrics.r2.toFixed(3)}</div>
                      </div>
                    )}
                    {metrics.mae !== undefined && metrics.mae !== null && (
                      <div>
                        <div className="text-[10px] text-neutral-400">MAE</div>
                        <div className="text-sm font-semibold text-indigo-300">{metrics.mae.toFixed(3)}</div>
                      </div>
                    )}
                  </div>
                </div>
              );
            }

            if (tr.tool === "analyze_image_tool" && tr.result && typeof tr.result === "object") {
              const res = tr.result as Record<string, unknown>;
              const obs = (res.observations || []) as string[];
              return (
                <div key={idx} className="rounded-xl border border-[#282b33] bg-[#111318] p-3 text-neutral-200">
                  <div className="flex items-center gap-2 font-medium text-purple-300 mb-2">
                    <Eye size={14} />
                    <span>Multimodal Visual Observations</span>
                  </div>
                  {obs.length > 0 ? (
                    <ul className="list-disc list-inside space-y-1 text-neutral-300 mt-1">
                      {obs.map((o, i) => (
                        <li key={i}>{o}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-neutral-400">{String(res.answer || "Visual analysis completed.")}</p>
                  )}
                </div>
              );
            }

            return null;
          })}
        </div>
      )}
    </div>
  );
}
