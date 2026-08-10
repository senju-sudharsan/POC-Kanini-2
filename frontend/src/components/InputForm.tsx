import { useState } from "react";
import { Send, SquarePen, StopCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

interface InputFormProps { onSubmit: (value: string) => void; onCancel: () => void; isLoading: boolean; hasHistory: boolean; }
export function InputForm({ onSubmit, onCancel, isLoading, hasHistory }: InputFormProps) {
  const [value, setValue] = useState("");
  const submit = (event?: React.FormEvent) => { event?.preventDefault(); if (!value.trim()) return; onSubmit(value); setValue(""); };
  return <form onSubmit={submit} className="flex flex-col gap-2 p-3 pb-4"><div className="flex items-start rounded-3xl rounded-bl-sm bg-neutral-700 px-4 pt-3"><Textarea value={value} onChange={(event) => setValue(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) submit(); }} placeholder="Ask an enterprise question…" className="border-0 text-neutral-100 placeholder:text-neutral-500" rows={1} />{isLoading ? <Button type="button" variant="ghost" size="icon" className="text-red-400" onClick={onCancel}><StopCircle /></Button> : <Button type="submit" variant="ghost" size="icon" className="text-blue-400" disabled={!value.trim()}><Send /></Button>}</div>{hasHistory && <Button type="button" onClick={() => window.location.reload()} className="self-end"><SquarePen size={16} /> New conversation</Button>}</form>;
}
