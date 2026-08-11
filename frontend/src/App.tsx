import { useCallback, useEffect, useRef, useState } from "react";
import type { ProcessedEvent } from "@/components/ActivityTimeline";
import { ChatMessagesView } from "@/components/ChatMessagesView";
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { Button } from "@/components/ui/button";
import { sendChat, sendDocumentChat, type ChatMessage } from "@/lib/chat";

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [liveActivities, setLiveActivities] = useState<ProcessedEvent[]>([]);
  const [history, setHistory] = useState<Record<string, ProcessedEvent[]>>({});
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [documentId, setDocumentId] = useState<string | null>(null);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [pendingApproval, setPendingApproval] = useState<{ approvalId: string; reason: string } | null>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const viewport = scrollAreaRef.current?.querySelector("[data-radix-scroll-area-viewport]");
    if (viewport) viewport.scrollTop = viewport.scrollHeight;
  }, [messages]);

  const submit = useCallback(
    async (value: string, approvalDecision?: "approved" | "rejected") => {
      const userMessage: ChatMessage = {
        role: "user",
        content: approvalDecision ? `[Decision: ${approvalDecision.toUpperCase()}] ${value}` : value,
        id: crypto.randomUUID(),
      };
      const activities: ProcessedEvent[] = [
        { title: "Stateful Router", data: "Processing request through Phase 7 stateful graph." },
        { title: documentId ? "Document Retrieval" : "LangGraph Agent", data: documentId ? "Finding indexed PDF evidence." : "Orchestrating agent reflection and specialist execution." },
      ];
      const controller = new AbortController();
      controllerRef.current = controller;
      setMessages((current) => [...current, userMessage]);
      setLiveActivities(activities);
      setError(null);
      setIsLoading(true);
      setPendingApproval(null);

      try {
        if (documentId) {
          const assistantMessage = await sendDocumentChat(value, documentId, controller.signal);
          setMessages((current) => [...current, assistantMessage]);
          setHistory((current) => ({ ...current, [assistantMessage.id]: activities }));
        } else {
          const res = await sendChat([...messages, userMessage], threadId, approvalDecision, controller.signal);
          setThreadId(res.thread_id);
          setMessages((current) => [...current, res.message]);
          const itemActivities = (res.activities && res.activities.length) ? res.activities : activities;
          setHistory((current) => ({ ...current, [res.message.id]: itemActivities }));
          if (res.approval_required && res.approval_id) {
            setPendingApproval({ approvalId: res.approval_id, reason: res.approval_reason || "Human approval required." });
          }
        }
      } catch (requestError) {
        if ((requestError as Error).name !== "AbortError")
          setError(requestError instanceof Error ? requestError.message : "The request failed.");
      } finally {
        if (controllerRef.current === controller) controllerRef.current = null;
        setIsLoading(false);
        setLiveActivities([]);
      }
    },
    [messages, documentId, threadId]
  );

  const handleApprovalSubmit = (decision: "approved" | "rejected") => {
    submit(decision === "approved" ? "Proceed with approved operation." : "Cancel rejected operation.", decision);
  };

  const cancel = useCallback(() => controllerRef.current?.abort(), []);

  if (error)
    return (
      <div className="flex h-screen items-center justify-center bg-neutral-800 text-neutral-100">
        <div className="space-y-4 text-center">
          <p>{error}</p>
          <Button onClick={() => setError(null)}>Retry</Button>
        </div>
      </div>
    );

  return (
    <div className="h-screen bg-neutral-800 font-sans antialiased">
      <main className="mx-auto h-full w-full max-w-4xl flex flex-col justify-between">
        {pendingApproval && (
          <div className="m-4 p-4 rounded-lg bg-amber-950/80 border border-amber-500/50 text-amber-100 space-y-3">
            <div className="flex items-center space-x-2 font-medium">
              <span className="text-amber-400">⚠️</span>
              <span>Human-In-The-Loop Approval Required</span>
            </div>
            <p className="text-sm text-amber-200/90">{pendingApproval.reason}</p>
            <div className="flex space-x-3 pt-1">
              <Button className="bg-emerald-600 hover:bg-emerald-500 text-white" onClick={() => handleApprovalSubmit("approved")}>
                Approve Operation
              </Button>
              <Button className="bg-amber-900/80 hover:bg-amber-800 text-amber-200 border border-amber-600" onClick={() => handleApprovalSubmit("rejected")}>
                Reject Operation
              </Button>
            </div>
          </div>
        )}
        {messages.length ? (
          <ChatMessagesView
            messages={messages}
            isLoading={isLoading}
            scrollAreaRef={scrollAreaRef}
            onSubmit={(val) => submit(val)}
            onCancel={cancel}
            liveActivityEvents={liveActivities}
            historicalActivities={history}
          />
        ) : (
          <WelcomeScreen onSubmit={(val) => submit(val)} onCancel={cancel} isLoading={isLoading} onIndexed={setDocumentId} />
        )}
      </main>
    </div>
  );
}
