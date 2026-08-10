import { Activity, ChevronDown, ChevronUp, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";

export interface ProcessedEvent { title: string; data: unknown; }
export function ActivityTimeline({ processedEvents, isLoading }: { processedEvents: ProcessedEvent[]; isLoading: boolean }) {
  const [collapsed, setCollapsed] = useState(false);
  useEffect(() => { if (!isLoading && processedEvents.length) setCollapsed(true); }, [isLoading, processedEvents.length]);
  return <Card className="max-h-96 bg-neutral-700"><CardHeader><CardDescription className="flex cursor-pointer items-center gap-2 text-neutral-100" onClick={() => setCollapsed(!collapsed)}>Activity {collapsed ? <ChevronDown size={16} /> : <ChevronUp size={16} />}</CardDescription></CardHeader>{!collapsed && <ScrollArea className="max-h-80"><CardContent>{processedEvents.map((event, index) => <div className="relative flex gap-3 pb-4" key={`${event.title}-${index}`}><Activity className="mt-0.5 size-4 shrink-0 text-neutral-300" /><div><p className="text-sm font-medium text-neutral-100">{event.title}</p><p className="text-xs text-neutral-300">{typeof event.data === "string" ? event.data : JSON.stringify(event.data)}</p></div></div>)}{isLoading && <div className="flex items-center gap-2 text-sm text-neutral-300"><Loader2 className="size-4 animate-spin" /> Processing…</div>}{!isLoading && !processedEvents.length && <p className="text-sm text-neutral-400">No activity to display.</p>}</CardContent></ScrollArea>}</Card>;
}
