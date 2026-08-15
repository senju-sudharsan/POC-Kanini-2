export type ChatMessage = { role: "user" | "assistant"; content: string; id: string };

export type ActivityEvent = { title: string; data: string };

export type ImageAttachment = { filename: string; mime_type: string; data: string };

export type CsvAttachment = { filename: string; data: string };

export type Citation = { label: string; filename?: string; page_number?: number; chunk_id?: string };

export type ToolResult = { tool: string; result?: unknown; error?: string };

export type ReportSection = { title: string; content: string; bullet_points?: string[] };

export type ReportPayload = {
  report_type: string;
  title: string;
  summary: string;
  sections: ReportSection[];
  metrics?: Record<string, unknown>;
  citations?: string[];
  recommendations?: string[];
};

export type ActionResult = { action_type: string; status: string; summary: string; metadata?: Record<string, unknown> };
export type SynthesisStatus = "success" | "degraded" | "quota_exhausted";

export type ChatResponse = {
  message: { role: "assistant"; content: string };
  thread_id: string;
  approval_required?: boolean;
  approval_id?: string | null;
  approval_reason?: string | null;
  operation?: string | null;
  activities?: ActivityEvent[];
  citations?: Citation[];
  tool_results?: ToolResult[];
  warnings?: string[];
  synthesis_status?: SynthesisStatus;
  reports?: ReportPayload[];
  actions?: ActionResult[];
};

type DocumentChatResponse = { answer: string; citations: { label: string }[] };

export async function sendChat(
  messages: ChatMessage[],
  threadId?: string | null,
  approval?: "approved" | "rejected" | null,
  signal?: AbortSignal,
  documentId?: string | null,
  attachments?: ImageAttachment[],
  csvData?: string | null
): Promise<ChatResponse & { message: ChatMessage }> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: messages.map(({ role, content }) => ({ role, content })),
      thread_id: threadId || undefined,
      document_id: documentId || null,
      document_ids: documentId ? [documentId] : [],
      attachments: attachments && attachments.length ? attachments : undefined,
      approval: approval || undefined,
      csv_data: csvData || undefined,
    }),
    signal,
  });
  const body = (await response.json().catch(() => null)) as ChatResponse | { detail?: string } | null;
  if (!response.ok)
    throw new Error(body && "detail" in body ? body.detail ?? "The assistant is unavailable." : "The assistant is unavailable.");
  if (!body || !("message" in body)) throw new Error("The assistant returned an invalid response.");

  return {
    ...body,
    message: { ...body.message, id: crypto.randomUUID() },
  };
}

export async function submitApprovalDecision(
  threadId: string,
  decision: "approved" | "rejected",
  approvalId?: string | null,
  message?: string,
  signal?: AbortSignal
): Promise<ChatResponse & { message: ChatMessage }> {
  const response = await fetch("/api/chat/approval", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      thread_id: threadId,
      decision,
      approval_id: approvalId || undefined,
      message: message || undefined,
    }),
    signal,
  });
  const body = (await response.json().catch(() => null)) as ChatResponse | { detail?: string } | null;
  if (!response.ok)
    throw new Error(body && "detail" in body ? body.detail ?? "Failed to submit approval." : "Failed to submit approval.");
  if (!body || !("message" in body)) throw new Error("Approval response was invalid.");

  return {
    ...body,
    message: { ...body.message, id: crypto.randomUUID() },
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

