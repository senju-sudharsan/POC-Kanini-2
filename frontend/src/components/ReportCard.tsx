import { useState } from "react";
import { FileSpreadsheet, ChevronDown, ChevronUp, CheckCircle, Lightbulb } from "lucide-react";
import type { ReportPayload } from "@/lib/chat";

interface Props {
  reports?: ReportPayload[];
}

export function ReportCard({ reports }: Props) {
  const [collapsed, setCollapsed] = useState(false);

  if (!reports || reports.length === 0) return null;

  return (
    <div className="mt-4 space-y-3">
      {reports.map((report, idx) => (
        <div key={idx} className="rounded-xl border border-[#282b33] bg-[#111318] text-neutral-100 p-3.5 shadow-lg">
          <div
            className="flex cursor-pointer items-center justify-between border-b border-[#282b33] pb-2.5"
            onClick={() => setCollapsed(!collapsed)}
          >
            <div className="flex items-center gap-2">
              <FileSpreadsheet className="text-indigo-400 size-4" />
              <span className="font-semibold text-sm text-neutral-100">{report.title}</span>
              <span className="rounded bg-[#171920] px-2 py-0.5 text-[10px] uppercase tracking-wider text-indigo-300 border border-[#282b33]">
                {report.report_type.replace("_", " ")}
              </span>
            </div>
            <button className="text-neutral-400 hover:text-neutral-200">
              {collapsed ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
            </button>
          </div>

          {!collapsed && (
            <div className="mt-3 space-y-3 text-xs">
              <p className="text-neutral-300 italic">{report.summary}</p>

              {report.sections && report.sections.length > 0 && (
                <div className="space-y-2">
                  {report.sections.map((sec, sIdx) => (
                    <div key={sIdx} className="rounded-lg bg-[#171920] p-2.5 border border-[#282b33]">
                      <h4 className="font-medium text-neutral-200 mb-1">{sec.title}</h4>
                      <p className="text-neutral-400 mb-1.5">{sec.content}</p>
                      {sec.bullet_points && sec.bullet_points.length > 0 && (
                        <ul className="list-disc list-inside space-y-0.5 text-neutral-300 pl-1">
                          {sec.bullet_points.map((bp, bIdx) => (
                            <li key={bIdx}>{bp}</li>
                          ))}
                        </ul>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {report.recommendations && report.recommendations.length > 0 && (
                <div className="rounded-lg bg-indigo-950/30 border border-indigo-500/30 p-2.5 text-indigo-200">
                  <div className="flex items-center gap-1.5 font-medium text-indigo-300 mb-1.5">
                    <Lightbulb size={14} />
                    <span>Recommendations</span>
                  </div>
                  <ul className="space-y-1">
                    {report.recommendations.map((rec, rIdx) => (
                      <li key={rIdx} className="flex items-start gap-1.5 text-neutral-300">
                        <CheckCircle size={12} className="text-indigo-400 mt-0.5 shrink-0" />
                        <span>{rec}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
