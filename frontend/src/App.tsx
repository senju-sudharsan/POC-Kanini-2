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
  const controllerRef = useRef<AbortController | null>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const viewport = scrollAreaRef.current?.querySelector("[data-radix-scroll-area-viewport]");
    if (viewport) viewport.scrollTop = viewport.scrollHeight;
  }, [messages]);

  const submit = useCallback(async (value: string) => {
    const userMessage: ChatMessage = { role: "user", content: value, id: crypto.randomUUID() };
    const activities: ProcessedEvent[] = [
      { title: "Understanding request", data: "Preparing the conversation context." },
      { title: documentId ? "Document retrieval" : "Gemini", data: documentId ? "Finding indexed PDF evidence and generating a grounded response." : "Generating a text response." },
    ];
    const controller = new AbortController();
    controllerRef.current = controller;
    setMessages((current) => [...current, userMessage]);
    setLiveActivities(activities);
    setError(null);
    setIsLoading(true);
    try {
      const assistantMessage = documentId ? await sendDocumentChat(value, documentId, controller.signal) : await sendChat([...messages, userMessage], controller.signal);
      setMessages((current) => [...current, assistantMessage]);
      setHistory((current) => ({ ...current, [assistantMessage.id]: activities }));
    } catch (requestError) {
      if ((requestError as Error).name !== "AbortError") setError(requestError instanceof Error ? requestError.message : "The request failed.");
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null;
      setIsLoading(false);
      setLiveActivities([]);
    }
  }, [messages, documentId]);

  const cancel = useCallback(() => controllerRef.current?.abort(), []);
  if (error) return <div className="flex h-screen items-center justify-center bg-neutral-800 text-neutral-100"><div className="space-y-4 text-center"><p>{error}</p><Button onClick={() => setError(null)}>Retry</Button></div></div>;
  return <div className="h-screen bg-neutral-800 font-sans antialiased"><main className="mx-auto h-full w-full max-w-4xl">{messages.length ? <ChatMessagesView messages={messages} isLoading={isLoading} scrollAreaRef={scrollAreaRef} onSubmit={submit} onCancel={cancel} liveActivityEvents={liveActivities} historicalActivities={history} /> : <WelcomeScreen onSubmit={submit} onCancel={cancel} isLoading={isLoading} onIndexed={setDocumentId} />}</main></div>;
}
