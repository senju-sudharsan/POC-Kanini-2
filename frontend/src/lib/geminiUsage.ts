import type { SynthesisStatus } from "@/lib/chat";

export type GeminiUsagePresentation = {
  label: string;
  detail: string;
  tone: "available" | "warning" | "limited";
};

/**
 * This deliberately reports only application-observed state. Gemini does not
 * expose an authoritative remaining-quota value to this chat response.
 */
export function getGeminiUsagePresentation(status: SynthesisStatus | undefined): GeminiUsagePresentation {
  if (status === "quota_exhausted") {
    return {
      label: "Usage limit reached",
      detail: "Gemini API quota or rate limit has been exhausted.",
      tone: "limited",
    };
  }
  if (status === "degraded") {
    return {
      label: "Provider status unknown",
      detail: "AURA used a safe fallback for the latest response.",
      tone: "warning",
    };
  }
  return {
    label: "Available",
    detail: "Last provider status: OK",
    tone: "available",
  };
}
