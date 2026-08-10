import { InputForm } from "@/components/InputForm";
import { DocumentUpload } from "@/components/DocumentUpload";

export function WelcomeScreen({ onSubmit, onCancel, isLoading, onIndexed }: { onSubmit: (value: string) => void; onCancel: () => void; isLoading: boolean; onIndexed: (documentId: string) => void }) {
  return <div className="flex h-full w-full max-w-3xl flex-col items-center justify-center gap-4 px-4 text-center"><div><h1 className="mb-3 text-5xl font-semibold text-neutral-100">Welcome.</h1><p className="text-xl text-neutral-400">How can I help with your enterprise questions?</p></div><div className="mt-4 w-full"><InputForm onSubmit={onSubmit} onCancel={onCancel} isLoading={isLoading} hasHistory={false} /><DocumentUpload onIndexed={onIndexed} /></div><p className="text-xs text-neutral-500">Gemini chat and evidence-backed PDF Q&A are ready. Data and ML tools are planned next.</p></div>;
}
