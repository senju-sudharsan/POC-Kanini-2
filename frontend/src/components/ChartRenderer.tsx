import { useState } from "react";
import {
  BarChart3,
  LineChart as LineChartIcon,
  PieChart as PieChartIcon,
  ScatterChart,
  Table as TableIcon,
  TrendingUp,
  ChevronDown,
  ChevronUp,
  AlertCircle,
} from "lucide-react";
import type { VisualizationPayload, KpiCard } from "@/lib/chat";

interface Props {
  visualizations?: VisualizationPayload[];
}

const PALETTE = [
  "#865CFF", // Neon purple
  "#4D8DFF", // Electric blue
  "#00F0FF", // Neon cyan
  "#B388FF", // Soft violet
  "#FF6B8B", // Coral pink
  "#2ED573", // Emerald green
  "#FFA502", // Amber gold
  "#E056FD", // Magenta
];

export function ChartRenderer({ visualizations }: Props) {
  const [collapsed, setCollapsed] = useState(false);

  if (!visualizations || visualizations.length === 0) return null;

  return (
    <div className="mt-4 space-y-4">
      {visualizations.map((viz, idx) => (
        <div
          key={idx}
          className="rounded-xl border border-[#282b33] bg-[#111318] text-neutral-100 p-4 shadow-xl transition-all duration-200 hover:border-[#3a3f4d]"
        >
          {/* Header */}
          <div
            className="flex cursor-pointer items-center justify-between border-b border-[#282b33] pb-3"
            onClick={() => setCollapsed(!collapsed)}
          >
            <div className="flex items-center gap-2.5">
              <ChartTypeIcon type={viz.chart_type} />
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-sm text-neutral-100">{viz.title}</span>
                  <span className="rounded bg-[#171920] px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-purple-300 border border-[#282b33]">
                    {viz.chart_type}
                  </span>
                </div>
                {viz.description && (
                  <p className="text-xs text-neutral-400 mt-0.5">{viz.description}</p>
                )}
              </div>
            </div>
            <button className="text-neutral-400 hover:text-neutral-200 transition-colors">
              {collapsed ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
            </button>
          </div>

          {/* Body */}
          {!collapsed && (
            <div className="mt-4">
              {viz.error ? (
                <div className="flex items-center gap-2 rounded-lg bg-red-950/30 border border-red-500/30 p-3 text-xs text-red-300">
                  <AlertCircle size={15} className="shrink-0 text-red-400" />
                  <span>{viz.error}</span>
                </div>
              ) : (
                <RenderChartBody viz={viz} />
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function ChartTypeIcon({ type }: { type: string }) {
  switch (type) {
    case "bar":
      return <BarChart3 className="text-purple-400 size-4 shrink-0" />;
    case "line":
      return <LineChartIcon className="text-blue-400 size-4 shrink-0" />;
    case "pie":
    case "donut":
      return <PieChartIcon className="text-pink-400 size-4 shrink-0" />;
    case "scatter":
      return <ScatterChart className="text-cyan-400 size-4 shrink-0" />;
    case "table":
      return <TableIcon className="text-amber-400 size-4 shrink-0" />;
    case "kpi":
      return <TrendingUp className="text-emerald-400 size-4 shrink-0" />;
    default:
      return <BarChart3 className="text-purple-400 size-4 shrink-0" />;
  }
}

function RenderChartBody({ viz }: { viz: VisualizationPayload }) {
  switch (viz.chart_type) {
    case "kpi":
      return <KpiCardsRenderer kpis={viz.kpis || []} />;
    case "bar":
      return <BarChartRenderer data={viz.data || []} xField={viz.x_field} yField={viz.y_field} />;
    case "line":
      return <LineChartRenderer data={viz.data || []} xField={viz.x_field} yField={viz.y_field} />;
    case "pie":
    case "donut":
      return <PieChartRenderer data={viz.data || []} isDonut={viz.chart_type === "donut"} />;
    case "scatter":
      return <ScatterPlotRenderer data={viz.data || []} xField={viz.x_field} yField={viz.y_field} />;
    case "table":
    default:
      return <TableRenderer data={viz.data || []} columns={viz.columns} />;
  }
}

/* ========================================================================= */
/* KPI Cards Renderer                                                        */
/* ========================================================================= */
function KpiCardsRenderer({ kpis }: { kpis: KpiCard[] }) {
  if (!kpis.length) return <p className="text-xs text-neutral-400">No KPI metrics available.</p>;

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
      {kpis.map((kpi, idx) => (
        <div
          key={idx}
          className="relative overflow-hidden rounded-xl border border-[#282b33] bg-[#171920] p-3.5 transition-all duration-200 hover:border-purple-500/40 hover:shadow-[0_0_15px_rgba(134,92,255,0.1)]"
        >
          <div className="text-[11px] font-medium text-neutral-400 uppercase tracking-wider">{kpi.label}</div>
          <div className="mt-1.5 text-lg font-bold text-neutral-100 tracking-tight">{String(kpi.value)}</div>
          {kpi.subtext && <div className="mt-1 text-[11px] text-purple-300/80">{kpi.subtext}</div>}
        </div>
      ))}
    </div>
  );
}

/* ========================================================================= */
/* Native SVG Bar Chart                                                      */
/* ========================================================================= */
function BarChartRenderer({
  data,
  xField,
  yField,
}: {
  data: Record<string, any>[];
  xField?: string | null;
  yField?: string | null;
}) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  if (!data || data.length === 0) {
    return <p className="text-xs text-neutral-400">No data points available for bar chart.</p>;
  }

  const values = data.map((d) => (typeof d.value === "number" ? d.value : Number(d.value) || 0));
  const maxVal = Math.max(...values, 1);
  const labels = data.map((d) => String(d.label ?? (xField ? d[xField] : "") ?? ""));

  const height = 220;
  const paddingLeft = 40;
  const paddingBottom = 45;
  const paddingTop = 20;
  const paddingRight = 20;
  const chartHeight = height - paddingTop - paddingBottom;

  return (
    <div className="w-full">
      <div className="relative w-full overflow-x-auto">
        <svg
          viewBox={`0 0 500 ${height}`}
          className="w-full h-auto min-w-[400px] text-xs select-none"
          style={{ maxHeight: "260px" }}
        >
          <defs>
            <linearGradient id="barGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#865CFF" />
              <stop offset="100%" stopColor="#4D8DFF" />
            </linearGradient>
            <linearGradient id="barGradientHover" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#B388FF" />
              <stop offset="100%" stopColor="#00F0FF" />
            </linearGradient>
          </defs>

          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((ratio, i) => {
            const y = paddingTop + chartHeight * (1 - ratio);
            const val = (maxVal * ratio).toFixed(ratio === 1 || maxVal >= 10 ? 0 : 1);
            return (
              <g key={i}>
                <line
                  x1={paddingLeft}
                  y1={y}
                  x2={500 - paddingRight}
                  y2={y}
                  stroke="#282b33"
                  strokeDasharray="3 3"
                  strokeWidth="1"
                />
                <text
                  x={paddingLeft - 8}
                  y={y + 3}
                  fill="#717684"
                  fontSize="10"
                  textAnchor="end"
                  fontFamily="sans-serif"
                >
                  {val}
                </text>
              </g>
            );
          })}

          {/* Bars */}
          {data.map((d, i) => {
            const numBars = data.length;
            const availableWidth = 500 - paddingLeft - paddingRight;
            const barSlotWidth = availableWidth / numBars;
            const barWidth = Math.min(Math.max(barSlotWidth * 0.65, 12), 40);
            const x = paddingLeft + i * barSlotWidth + (barSlotWidth - barWidth) / 2;
            const val = typeof d.value === "number" ? d.value : Number(d.value) || 0;
            const barHeight = (val / maxVal) * chartHeight;
            const y = paddingTop + chartHeight - barHeight;
            const isHovered = hoveredIdx === i;

            return (
              <g
                key={i}
                onMouseEnter={() => setHoveredIdx(i)}
                onMouseLeave={() => setHoveredIdx(null)}
                className="cursor-pointer transition-all duration-150"
              >
                <rect
                  x={x}
                  y={y}
                  width={barWidth}
                  height={Math.max(barHeight, 2)}
                  rx="4"
                  fill={isHovered ? "url(#barGradientHover)" : "url(#barGradient)"}
                  opacity={hoveredIdx === null || isHovered ? 1 : 0.6}
                />
                {/* Bar top value label */}
                {isHovered && (
                  <text
                    x={x + barWidth / 2}
                    y={y - 6}
                    fill="#00F0FF"
                    fontSize="10"
                    fontWeight="bold"
                    textAnchor="middle"
                  >
                    {val}
                  </text>
                )}
                {/* X axis category label */}
                <text
                  x={x + barWidth / 2}
                  y={height - paddingBottom + 16}
                  fill={isHovered ? "#B388FF" : "#9E9E9E"}
                  fontSize="9.5"
                  textAnchor="middle"
                  transform={`rotate(-25, ${x + barWidth / 2}, ${height - paddingBottom + 16})`}
                  className="truncate"
                >
                  {labels[i].length > 12 ? labels[i].slice(0, 10) + "…" : labels[i]}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Footer Axis Description */}
      <div className="mt-2 flex justify-between text-[11px] text-neutral-400 px-1 border-t border-[#1e2128] pt-2">
        <span>Grouping: <strong className="text-neutral-300">{xField || "Category"}</strong></span>
        <span>Metric: <strong className="text-neutral-300">{yField || "Value"}</strong></span>
      </div>
    </div>
  );
}

/* ========================================================================= */
/* Native SVG Line Chart                                                     */
/* ========================================================================= */
function LineChartRenderer({
  data,
  xField,
  yField,
}: {
  data: Record<string, any>[];
  xField?: string | null;
  yField?: string | null;
}) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  if (!data || data.length === 0) {
    return <p className="text-xs text-neutral-400">No data points available for line chart.</p>;
  }

  const values = data.map((d) => (typeof d.value === "number" ? d.value : Number(d.value) || 0));
  const minVal = Math.min(...values);
  const maxVal = Math.max(...values, 1);
  const valRange = maxVal - minVal || 1;
  const labels = data.map((d) => String(d.label ?? (xField ? d[xField] : "") ?? ""));

  const height = 220;
  const paddingLeft = 40;
  const paddingBottom = 40;
  const paddingTop = 20;
  const paddingRight = 20;
  const chartHeight = height - paddingTop - paddingBottom;
  const availableWidth = 500 - paddingLeft - paddingRight;

  const points = data.map((d, i) => {
    const x = paddingLeft + (i / Math.max(data.length - 1, 1)) * availableWidth;
    const val = typeof d.value === "number" ? d.value : Number(d.value) || 0;
    const y = paddingTop + chartHeight - ((val - minVal) / valRange) * chartHeight;
    return { x, y, val, label: labels[i] };
  });

  const pathD = points.reduce((acc, pt, i) => `${acc} ${i === 0 ? "M" : "L"} ${pt.x},${pt.y}`, "");
  const areaD = `${pathD} L ${points[points.length - 1].x},${paddingTop + chartHeight} L ${points[0].x},${paddingTop + chartHeight} Z`;

  return (
    <div className="w-full">
      <div className="relative w-full overflow-x-auto">
        <svg
          viewBox={`0 0 500 ${height}`}
          className="w-full h-auto min-w-[400px] text-xs select-none"
          style={{ maxHeight: "260px" }}
        >
          <defs>
            <linearGradient id="lineAreaGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#865CFF" stopOpacity="0.3" />
              <stop offset="100%" stopColor="#4D8DFF" stopOpacity="0.0" />
            </linearGradient>
          </defs>

          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((ratio, i) => {
            const y = paddingTop + chartHeight * (1 - ratio);
            const val = (minVal + valRange * ratio).toFixed(ratio === 1 || maxVal >= 10 ? 0 : 1);
            return (
              <g key={i}>
                <line
                  x1={paddingLeft}
                  y1={y}
                  x2={500 - paddingRight}
                  y2={y}
                  stroke="#282b33"
                  strokeDasharray="3 3"
                  strokeWidth="1"
                />
                <text
                  x={paddingLeft - 8}
                  y={y + 3}
                  fill="#717684"
                  fontSize="10"
                  textAnchor="end"
                >
                  {val}
                </text>
              </g>
            );
          })}

          {/* Area Fill */}
          <path d={areaD} fill="url(#lineAreaGradient)" />

          {/* Line Path */}
          <path d={pathD} fill="none" stroke="#865CFF" strokeWidth="2.5" strokeLinecap="round" />

          {/* Points */}
          {points.map((pt, i) => {
            const isHovered = hoveredIdx === i;
            return (
              <g
                key={i}
                onMouseEnter={() => setHoveredIdx(i)}
                onMouseLeave={() => setHoveredIdx(null)}
                className="cursor-pointer"
              >
                <circle
                  cx={pt.x}
                  cy={pt.y}
                  r={isHovered ? 6 : 3.5}
                  fill={isHovered ? "#00F0FF" : "#4D8DFF"}
                  stroke="#111318"
                  strokeWidth="2"
                />
                {isHovered && (
                  <text
                    x={pt.x}
                    y={pt.y - 10}
                    fill="#00F0FF"
                    fontSize="10"
                    fontWeight="bold"
                    textAnchor="middle"
                  >
                    {pt.val}
                  </text>
                )}
                {/* Every Nth X-axis label */}
                {(i % Math.ceil(data.length / 6) === 0 || i === data.length - 1) && (
                  <text
                    x={pt.x}
                    y={height - paddingBottom + 16}
                    fill="#9E9E9E"
                    fontSize="9.5"
                    textAnchor="middle"
                  >
                    {pt.label.length > 10 ? pt.label.slice(0, 8) + "…" : pt.label}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>

      <div className="mt-2 flex justify-between text-[11px] text-neutral-400 px-1 border-t border-[#1e2128] pt-2">
        <span>Timeline / X: <strong className="text-neutral-300">{xField || "Index"}</strong></span>
        <span>Metric / Y: <strong className="text-neutral-300">{yField || "Value"}</strong></span>
      </div>
    </div>
  );
}

/* ========================================================================= */
/* Native SVG Pie / Donut Chart                                              */
/* ========================================================================= */
function PieChartRenderer({
  data,
  isDonut = false,
}: {
  data: Record<string, any>[];
  isDonut?: boolean;
}) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  if (!data || data.length === 0) {
    return <p className="text-xs text-neutral-400">No data points available for pie chart.</p>;
  }

  const values = data.map((d) => (typeof d.value === "number" ? d.value : Number(d.value) || 0));
  const total = values.reduce((sum, v) => sum + v, 0) || 1;

  const cx = 110;
  const cy = 110;
  const r = 90;
  const innerR = isDonut ? 52 : 0;

  let currentAngle = -Math.PI / 2;
  const slices = data.map((d, i) => {
    const val = typeof d.value === "number" ? d.value : Number(d.value) || 0;
    const sliceAngle = (val / total) * 2 * Math.PI;
    const startAngle = currentAngle;
    const endAngle = currentAngle + sliceAngle;
    currentAngle = endAngle;

    const x1 = cx + r * Math.cos(startAngle);
    const y1 = cy + r * Math.sin(startAngle);
    const x2 = cx + r * Math.cos(endAngle);
    const y2 = cy + r * Math.sin(endAngle);

    const x3 = cx + innerR * Math.cos(endAngle);
    const y3 = cy + innerR * Math.sin(endAngle);
    const x4 = cx + innerR * Math.cos(startAngle);
    const y4 = cy + innerR * Math.sin(startAngle);

    const largeArc = sliceAngle > Math.PI ? 1 : 0;

    const pathD = isDonut
      ? `M ${x1},${y1} A ${r},${r} 0 ${largeArc},1 ${x2},${y2} L ${x3},${y3} A ${innerR},${innerR} 0 ${largeArc},0 ${x4},${y4} Z`
      : `M ${cx},${cy} L ${x1},${y1} A ${r},${r} 0 ${largeArc},1 ${x2},${y2} Z`;

    const percentage = ((val / total) * 100).toFixed(1);
    const color = PALETTE[i % PALETTE.length];

    return { pathD, val, percentage, color, label: String(d.label || `Item ${i + 1}`) };
  });

  return (
    <div className="flex flex-col sm:flex-row items-center gap-6">
      {/* Donut SVG */}
      <div className="shrink-0 relative">
        <svg width="220" height="220" viewBox="0 0 220 220" className="select-none">
          {slices.map((slice, i) => (
            <path
              key={i}
              d={slice.pathD}
              fill={slice.color}
              opacity={hoveredIdx === null || hoveredIdx === i ? 1 : 0.45}
              stroke="#111318"
              strokeWidth="2"
              className="cursor-pointer transition-all duration-150"
              onMouseEnter={() => setHoveredIdx(i)}
              onMouseLeave={() => setHoveredIdx(null)}
            />
          ))}
          {isDonut && (
            <text
              x={cx}
              y={cy + 4}
              fill="#FFFFFF"
              fontSize="12"
              fontWeight="bold"
              textAnchor="middle"
            >
              {hoveredIdx !== null ? `${slices[hoveredIdx].percentage}%` : "Total"}
            </text>
          )}
        </svg>
      </div>

      {/* Legend */}
      <div className="flex flex-col gap-1.5 w-full text-xs">
        {slices.map((slice, i) => (
          <div
            key={i}
            onMouseEnter={() => setHoveredIdx(i)}
            onMouseLeave={() => setHoveredIdx(null)}
            className={`flex items-center justify-between p-1.5 rounded-md cursor-pointer transition-colors ${
              hoveredIdx === i ? "bg-[#1f222b]" : "hover:bg-[#171920]"
            }`}
          >
            <div className="flex items-center gap-2 truncate">
              <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: slice.color }} />
              <span className="truncate text-neutral-300 font-medium">{slice.label}</span>
            </div>
            <div className="flex items-center gap-2 text-neutral-400 font-mono text-[11px] shrink-0">
              <span>{slice.val}</span>
              <span className="text-purple-400 font-semibold">({slice.percentage}%)</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ========================================================================= */
/* Native SVG Scatter Plot                                                   */
/* ========================================================================= */
function ScatterPlotRenderer({
  data,
  xField,
  yField,
}: {
  data: Record<string, any>[];
  xField?: string | null;
  yField?: string | null;
}) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  if (!data || data.length === 0) {
    return <p className="text-xs text-neutral-400">No data points available for scatter plot.</p>;
  }

  const xVals = data.map((d) => Number(d.x ?? d[xField || "x"] ?? 0));
  const yVals = data.map((d) => Number(d.y ?? d[yField || "y"] ?? 0));

  const minX = Math.min(...xVals);
  const maxX = Math.max(...xVals, 1);
  const rangeX = maxX - minX || 1;

  const minY = Math.min(...yVals);
  const maxY = Math.max(...yVals, 1);
  const rangeY = maxY - minY || 1;

  const height = 220;
  const paddingLeft = 45;
  const paddingBottom = 40;
  const paddingTop = 20;
  const paddingRight = 20;
  const chartHeight = height - paddingTop - paddingBottom;
  const availableWidth = 500 - paddingLeft - paddingRight;

  return (
    <div className="w-full">
      <div className="relative w-full overflow-x-auto">
        <svg
          viewBox={`0 0 500 ${height}`}
          className="w-full h-auto min-w-[400px] text-xs select-none"
          style={{ maxHeight: "260px" }}
        >
          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((ratio, i) => {
            const y = paddingTop + chartHeight * (1 - ratio);
            const val = (minY + rangeY * ratio).toFixed(1);
            return (
              <g key={i}>
                <line
                  x1={paddingLeft}
                  y1={y}
                  x2={500 - paddingRight}
                  y2={y}
                  stroke="#282b33"
                  strokeDasharray="3 3"
                  strokeWidth="1"
                />
                <text
                  x={paddingLeft - 8}
                  y={y + 3}
                  fill="#717684"
                  fontSize="10"
                  textAnchor="end"
                >
                  {val}
                </text>
              </g>
            );
          })}

          {/* Scatter dots */}
          {data.map((d, i) => {
            const xVal = Number(d.x ?? d[xField || "x"] ?? 0);
            const yVal = Number(d.y ?? d[yField || "y"] ?? 0);
            const cx = paddingLeft + ((xVal - minX) / rangeX) * availableWidth;
            const cy = paddingTop + chartHeight - ((yVal - minY) / rangeY) * chartHeight;
            const isHovered = hoveredIdx === i;

            return (
              <g
                key={i}
                onMouseEnter={() => setHoveredIdx(i)}
                onMouseLeave={() => setHoveredIdx(null)}
                className="cursor-pointer"
              >
                <circle
                  cx={cx}
                  cy={cy}
                  r={isHovered ? 6.5 : 4}
                  fill={isHovered ? "#00F0FF" : "#865CFF"}
                  stroke="#111318"
                  strokeWidth="1.5"
                  opacity={hoveredIdx === null || isHovered ? 0.9 : 0.4}
                />
                {isHovered && (
                  <text
                    x={cx}
                    y={cy - 9}
                    fill="#00F0FF"
                    fontSize="10"
                    fontWeight="bold"
                    textAnchor="middle"
                  >
                    ({xVal}, {yVal})
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>

      <div className="mt-2 flex justify-between text-[11px] text-neutral-400 px-1 border-t border-[#1e2128] pt-2">
        <span>X Axis: <strong className="text-neutral-300">{xField || "X"}</strong></span>
        <span>Y Axis: <strong className="text-neutral-300">{yField || "Y"}</strong></span>
      </div>
    </div>
  );
}

/* ========================================================================= */
/* Tabular Data View Renderer                                                */
/* ========================================================================= */
function TableRenderer({
  data,
  columns,
}: {
  data: Record<string, any>[];
  columns?: string[];
}) {
  if (!data || data.length === 0) {
    return <p className="text-xs text-neutral-400">No tabular records to display.</p>;
  }

  const tableCols = columns && columns.length > 0 ? columns : Object.keys(data[0] || {});

  return (
    <div className="overflow-x-auto rounded-lg border border-[#282b33]">
      <table className="w-full text-left text-xs">
        <thead className="bg-[#171920] text-neutral-400 font-semibold border-b border-[#282b33]">
          <tr>
            {tableCols.map((col, idx) => (
              <th key={idx} className="px-3 py-2 uppercase tracking-wider text-[10px]">
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-[#282b33] bg-[#111318] text-neutral-300">
          {data.slice(0, 15).map((row, rIdx) => (
            <tr key={rIdx} className="hover:bg-[#171920]/60 transition-colors">
              {tableCols.map((col, cIdx) => (
                <td key={cIdx} className="px-3 py-2 font-mono text-[11px]">
                  {String(row[col] ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
