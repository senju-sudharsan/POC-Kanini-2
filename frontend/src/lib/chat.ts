export type ChatMessage = { role: "user" | "assistant"; content: string; id: string };

export type ActivityEvent = { title: string; data: string };

export type ChatResponse = {
  message: { role: "assistant"; content: string };
  thread_id: string;
  approval_required?: boolean;
  approval_id?: string | null;
  approval_reason?: string | null;
  activities?: ActivityEvent[];
};

type DocumentChatResponse = { answer: string; citations: { label: string }[] };

export async function sendChat(
  messages: ChatMessage[],
  threadId?: string | null,
  approval?: "approved" | "rejected" | null,
  signal?: AbortSignal
): Promise<{ message: ChatMessage; thread_id: string; approval_required?: boolean; approval_id?: string | null; approval_reason?: string | null; activities?: ActivityEvent[] }> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: messages.map(({ role, content }) => ({ role, content })),
      thread_id: threadId || undefined,
      approval: approval || undefined,
    }),
    signal,
  });
  const body = (await response.json().catch(() => null)) as ChatResponse | { detail?: string } | null;
  if (!response.ok)
    throw new Error(body && "detail" in body ? body.detail ?? "The assistant is unavailable." : "The assistant is unavailable.");
  if (!body || !("message" in body)) throw new Error("The assistant returned an invalid response.");

  return {
    message: { ...body.message, id: crypto.randomUUID() },
    thread_id: body.thread_id,
    approval_required: body.approval_required,
    approval_id: body.approval_id,
    approval_reason: body.approval_reason,
    activities: body.activities,
  };
}

export async function sendDocumentChat(
  question: string,
  documentId: string,
  signal?: AbortSignal
): Promise<ChatMessage> {
  const response = await fetch("/api/documents/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, document_id: documentId }),
    signal,
  });
  const body = (await response.json().catch(() => null)) as DocumentChatResponse | { detail?: string } | null;
  if (!response.ok)
    throw new Error(body && "detail" in body ? body.detail ?? "Document Q&A is unavailable." : "Document Q&A is unavailable.");
  if (!body || !("answer" in body)) throw new Error("Document Q&A returned an invalid response.");
  const references = body.citations.length
    ? `\n\nSources: ${body.citations.map((citation) => `[${citation.label}]`).join(" ")}`
    : "";
  return { role: "assistant", content: `${body.answer}${references}`, id: crypto.randomUUID() };
}
