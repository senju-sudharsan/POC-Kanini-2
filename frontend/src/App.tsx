import { useCallback, useEffect, useRef, useState } from "react";
import type { ProcessedEvent } from "@/components/ActivityTimeline";
import type { AuraState } from "@/components/AuraOrb";
import { ChatComposer } from "@/components/ChatComposer";
import { ChatMessagesView, type ExtendedChatMessage } from "@/components/ChatMessagesView";
import { Topbar } from "@/components/Topbar";
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { Button } from "@/components/ui/button";
import { sendChat, type CsvAttachment, type ImageAttachment, type SynthesisStatus } from "@/lib/chat";

export default function App() {
  const [messages, setMessages] = useState<ExtendedChatMessage[]>([]);
  const [liveActivities, setLiveActivities] = useState<ProcessedEvent[]>([]);
  const [history, setHistory] = useState<Record<string, ProcessedEvent[]>>({});
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isFocused, setIsFocused] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [documentId, setDocumentId] = useState<string | null>(null);
  const [documentName, setDocumentName] = useState<string | null>(null);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [pendingApproval, setPendingApproval] = useState<{ approvalId: string; reason: string; operation?: string } | null>(null);
  const [geminiRequestCount, setGeminiRequestCount] = useState<number>(() => {
    const saved = localStorage.getItem("aura_request_count");
    if (saved) {
      const parsed = parseInt(saved, 10);
      return isNaN(parsed) ? 0 : parsed;
    }
    return 0;
  });
  const [lastSynthesisStatus, setLastSynthesisStatus] = useState<SynthesisStatus | undefined>();
  const controllerRef = useRef<AbortController | null>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    localStorage.setItem("aura_request_count", String(geminiRequestCount));
  }, [geminiRequestCount]);

  useEffect(() => {
    const viewport = scrollAreaRef.current?.querySelector("[data-radix-scroll-area-viewport]");
    if (viewport) viewport.scrollTop = viewport.scrollHeight;
  }, [messages, pendingApproval]);

  // Compute dynamic AuraOrb state based on real application lifecycle
  const computeAuraState = (): AuraState => {
    if (pendingApproval) return "hitl";
    if (error) return "error";
    if (isLoading) {
      const lastAct = liveActivities.length ? liveActivities[liveActivities.length - 1].title.toLowerCase() : "";
      if (lastAct.includes("vision") || lastAct.includes("multimodal") || lastAct.includes("image")) return "vision";
      if (lastAct.includes("rag") || lastAct.includes("document") || lastAct.includes("evidence")) return "retrieval";
      if (lastAct.includes("ml") || lastAct.includes("profile") || lastAct.includes("dataset") || lastAct.includes("train")) return "ml";
      return "processing";
    }
    if (isTyping) return "typing";
    if (isFocused) return "focus";
    if (messages.length > 0) return "result";
    return "idle";
  };

  const auraState = computeAuraState();

  const submit = useCallback(
    async (value: string, attachments?: ImageAttachment[], csvAtt?: CsvAttachment, approvalDecision?: "approved" | "rejected") => {
      const userMessage: ExtendedChatMessage = {
        role: "user",
        content: approvalDecision ? `[Decision: ${approvalDecision.toUpperCase()}] ${value}` : value,
        id: crypto.randomUUID(),
        attachments: attachments,
      };

      const activities: ProcessedEvent[] = [
        { title: "Supervisor Routing", data: "Analyzing request context and selecting specialist workflow." },
        {
          title: attachments?.length
            ? "Multimodal Vision"
            : csvAtt
            ? "Dataset Profiling"
            : documentId
            ? "Document RAG Evidence"
            : "General Conversation",
          data: attachments?.length
            ? `Analyzing ${attachments.length} image attachment(s).`
            : csvAtt
            ? `Profiling dataset: ${csvAtt.filename}.`
            : documentId
            ? `Searching indexed document evidence (${documentName || documentId}).`
            : "Synthesizing executive response.",
        },
      ];

      const controller = new AbortController();
      controllerRef.current = controller;
      setMessages((current) => [...current, userMessage]);
      setLiveActivities(activities);
      setError(null);
      setIsLoading(true);
      setPendingApproval(null);

      try {
        const res = await sendChat(
          [...messages, userMessage],
          threadId,
          approvalDecision,
          controller.signal,
          documentId,
          attachments,
          csvAtt?.data ?? null
        );

        setThreadId(res.thread_id);
        setGeminiRequestCount((count) => count + 1);
        setLastSynthesisStatus(res.synthesis_status);

        const assistantMsg: ExtendedChatMessage = {
          ...res.message,
          citations: res.citations,
          toolResults: res.tool_results,
          warnings: res.warnings,
          synthesisStatus: res.synthesis_status,
          reports: res.reports,
          approvalRequired: res.approval_required,
          approvalId: res.approval_id || undefined,
          approvalReason: res.approval_reason || undefined,
          operation: res.operation || undefined,
          approvalDecision: approvalDecision || undefined,
        };

        setMessages((current) => [...current, assistantMsg]);
        const itemActivities = res.activities && res.activities.length ? res.activities : activities;
        setHistory((current) => ({ ...current, [assistantMsg.id]: itemActivities }));

        if (res.approval_required && res.approval_id) {
          setPendingApproval({
            approvalId: res.approval_id,
            reason: res.approval_reason || "Human authorization required before proceeding.",
            operation: res.operation || "ml",
          });
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
    [messages, documentId, documentName, threadId]
  );

  const handleApprovalSubmit = (decision: "approved" | "rejected") => {
    submit(
      decision === "approved" ? "Proceed with approved operation." : "Cancel rejected operation.",
      undefined,
      undefined,
      decision
    );
  };

  const handleDocumentIndexed = (docId: string, filename?: string) => {
    setDocumentId(docId);
    if (filename) setDocumentName(filename);
  };

  const handleUnlinkDocument = useCallback(() => {
    setDocumentId(null);
    setDocumentName(null);
  }, []);

  const handleResetChat = () => {
    setMessages([]);
    setLiveActivities([]);
    setHistory({});
    setError(null);
    setThreadId(null);
    setPendingApproval(null);
    setGeminiRequestCount(0);
    setLastSynthesisStatus(undefined);
  };

  const cancel = useCallback(() => controllerRef.current?.abort(), []);

  return (
    <div className="aura-app">
      <Topbar
        documentName={documentName}
        onResetChat={handleResetChat}
        onUnlinkDocument={handleUnlinkDocument}
        geminiRequestCount={geminiRequestCount}
        lastSynthesisStatus={lastSynthesisStatus}
      />

      <main className="conversation-container">
        {error && (
          <div className="m-4 p-4 rounded-xl bg-red-950/80 border border-red-800 text-red-200 flex items-center justify-between text-xs">
            <span>{error}</span>
            <Button variant="ghost" onClick={() => setError(null)} className="h-7 text-xs px-2 text-red-300 hover:text-white">
              Dismiss
            </Button>
          </div>
        )}

        {messages.length === 0 ? (
          <WelcomeScreen
            onIndexed={handleDocumentIndexed}
            documentName={documentName}
            onUnlinkDocument={handleUnlinkDocument}
            auraState={auraState}
            onPromptSelect={(prompt) => submit(prompt)}
          />
        ) : (
          <ChatMessagesView
            messages={messages}
            isLoading={isLoading}
            scrollAreaRef={scrollAreaRef}
            liveActivityEvents={liveActivities}
            historicalActivities={history}
            pendingApproval={pendingApproval}
            onApprovalDecision={handleApprovalSubmit}
          />
        )}
      </main>

      <ChatComposer
        onSubmit={(val, atts, csvAtt) => submit(val, atts, csvAtt)}
        onCancel={cancel}
        isLoading={isLoading}
        onFocusChange={setIsFocused}
        onTypingChange={setIsTyping}
        onIndexed={handleDocumentIndexed}
        onUnlinkDocument={handleUnlinkDocument}
        documentName={documentName}
        auraState={auraState}
      />
    </div>
  );
}
