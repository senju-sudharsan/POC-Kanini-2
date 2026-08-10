import type React from "react";
import { Copy, CopyCheck, Loader2 } from "lucide-react";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { ActivityTimeline, type ProcessedEvent } from "@/components/ActivityTimeline";
import { InputForm } from "@/components/InputForm";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { ChatMessage } from "@/lib/chat";

interface Props { messages: ChatMessage[]; isLoading: boolean; scrollAreaRef: React.RefObject<HTMLDivElement | null>; onSubmit: (value: string) => void; onCancel: () => void; liveActivityEvents: ProcessedEvent[]; historicalActivities: Record<string, ProcessedEvent[]>; }

export function ChatMessagesView({ messages, isLoading, scrollAreaRef, onSubmit, onCancel, liveActivityEvents, historicalActivities }: Props) {
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const copy = async (value: string, id: string) => { await navigator.clipboard.writeText(value); setCopiedId(id); window.setTimeout(() => setCopiedId(null), 2000); };
  return <div className="flex h-full flex-col"><ScrollArea className="flex-1 overflow-y-auto" ref={scrollAreaRef}><div className="mx-auto max-w-4xl space-y-3 p-4 pt-16 md:p-6">{messages.map((message, index) => { const isHuman = message.role === "user"; const isLast = index === messages.length - 1; const activities = isLast && isLoading ? liveActivityEvents : historicalActivities[message.id]; return <div key={message.id} className={isHuman ? "flex justify-end" : "flex flex-col"}>{!isHuman && activities?.length ? <div className="mb-3"><ActivityTimeline processedEvents={activities} isLoading={isLast && isLoading} /></div> : null}<div className={isHuman ? "max-w-[90%] rounded-3xl rounded-br-lg bg-neutral-700 px-4 pt-3 text-white" : "break-words text-neutral-100"}><ReactMarkdown>{message.content}</ReactMarkdown></div>{!isHuman && message.content && <Button variant="ghost" className="mt-2 self-end text-neutral-300" onClick={() => copy(message.content, message.id)}>{copiedId === message.id ? <CopyCheck /> : <Copy />}{copiedId === message.id ? "Copied" : "Copy"}</Button>}</div>; })}{isLoading && messages.at(-1)?.role === "user" && <div className="rounded-lg bg-neutral-800 p-3 text-neutral-100">{liveActivityEvents.length ? <ActivityTimeline processedEvents={liveActivityEvents} isLoading /> : <span className="flex items-center gap-2"><Loader2 className="animate-spin" size={18} /> Processing…</span>}</div>}</div></ScrollArea><InputForm onSubmit={onSubmit} onCancel={onCancel} isLoading={isLoading} hasHistory={messages.length > 0} /></div>;
}
