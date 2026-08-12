import type React from "react";
import { Check, Copy, CopyCheck } from "lucide-react";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import type { ProcessedEvent } from "@/components/ActivityTimeline";
import { ToolResultCard } from "@/components/ToolResultCard";
import { ReportCard } from "@/components/ReportCard";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { ChatMessage, Citation, ToolResult, ReportPayload, ImageAttachment } from "@/lib/chat";

export interface ExtendedChatMessage extends ChatMessage {
  citations?: Citation[];
  toolResults?: ToolResult[];
  warnings?: string[];
  reports?: ReportPayload[];
  attachments?: ImageAttachment[];
}

interface Props {
  messages: ExtendedChatMessage[];
  isLoading: boolean;
  scrollAreaRef: React.RefObject<HTMLDivElement | null>;
  liveActivityEvents: ProcessedEvent[];
  historicalActivities: Record<string, ProcessedEvent[]>;
}

// Convert technical graph node names to friendly human-readable phrases
function sanitizeActivityTitle(raw: string): string {
  const title = raw.toLowerCase();
  if (title.includes("supervisor") || title.includes("hybrid router")) return "Analyzing request context...";
  if (title.includes("rag") || title.includes("document")) return "Searching document evidence...";
  if (title.includes("multimodal") || title.includes("vision")) return "Analyzing visual attachment...";
  if (title.includes("profile") || title.includes("dataset")) return "Profiling dataset structure...";
  if (title.includes("train") || title.includes("ml")) return "Training machine learning model...";
  if (title.includes("predict")) return "Generating model predictions...";
  if (title.includes("synthesis") || title.includes("report")) return "Synthesizing executive response...";
  if (title.includes("hitl") || title.includes("approval")) return "Waiting for authorization...";
  return raw;
}

export function ChatMessagesView({
  messages,
  isLoading,
  scrollAreaRef,
  liveActivityEvents,
  historicalActivities,
}: Props) {
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const copy = async (value: string, id: string) => {
    await navigator.clipboard.writeText(value);
    setCopiedId(id);
    window.setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <ScrollArea className="flex-1 overflow-y-auto" ref={scrollAreaRef}>
      <div className="thread max-w-3xl mx-auto px-4 pt-6 pb-40">
        {messages.map((message, index) => {
          const isHuman = message.role === "user";
          const isLast = index === messages.length - 1;
          const activities = isLast && isLoading ? liveActivityEvents : historicalActivities[message.id];

          if (isHuman) {
            return (
              <article key={message.id} className="user-turn">
                <div className="turn-meta">
                  You <time>Just now</time>
                </div>
                <p>{message.content}</p>
                {message.attachments && message.attachments.length > 0 && (
                  <div className="flex flex-wrap gap-2.5 mt-3">
                    {message.attachments.map((att, aIdx) => (
                      <div
                        key={aIdx}
                        className="flex items-center gap-2 border border-[#262a32] rounded-xl p-2 bg-[#121419]/80 max-w-xs"
                      >
                        <img
                          src={`data:${att.mime_type};base64,${att.data}`}
                          alt={att.filename}
                          className="h-12 w-12 rounded-lg object-cover"
                        />
                        <div className="overflow-hidden text-xs">
                          <div className="font-medium text-neutral-200 truncate">{att.filename}</div>
                          <div className="text-neutral-500 text-[10px] uppercase">{att.mime_type.split("/")[1]}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </article>
            );
          }

          return (
            <article key={message.id} className="assistant-turn">
              <div className="assistant-meta">
                <span className="mini-orb" /> AURA
              </div>

              {activities && activities.length > 0 && (
                <div className="tool-trace">
                  <span className="trace-pulse" />
                  <span>{sanitizeActivityTitle(activities[activities.length - 1].title)}</span>
                  <Check className="w-3.5 h-3.5 text-neutral-500 ml-auto" />
                </div>
              )}

              <div className="prose prose-invert max-w-none text-neutral-200 text-sm md:text-base leading-relaxed">
                <ReactMarkdown>{message.content}</ReactMarkdown>
              </div>

              <ToolResultCard
                citations={message.citations}
                toolResults={message.toolResults}
                warnings={message.warnings}
              />
              <ReportCard reports={message.reports} />

              <div className="flex items-center justify-end mt-2">
                <Button
                  variant="ghost"
                  className="text-neutral-500 hover:text-neutral-300 text-xs h-7 px-2 gap-1"
                  onClick={() => copy(message.content, message.id)}
                >
                  {copiedId === message.id ? <CopyCheck size={13} /> : <Copy size={13} />}
                  {copiedId === message.id ? "Copied" : "Copy"}
                </Button>
              </div>
            </article>
          );
        })}

        {isLoading && messages.at(-1)?.role === "user" && (
          <article className="assistant-turn animate-pulse">
            <div className="assistant-meta">
              <span className="mini-orb animate-pulse" /> AURA
            </div>
            <div className="thinking flex items-center gap-2 text-xs text-neutral-400">
              <span />
              <span />
              <span />
              <span>
                {liveActivityEvents.length
                  ? sanitizeActivityTitle(liveActivityEvents[liveActivityEvents.length - 1].title)
                  : "Connecting signals..."}
              </span>
            </div>
          </article>
        )}
      </div>
    </ScrollArea>
  );
}
