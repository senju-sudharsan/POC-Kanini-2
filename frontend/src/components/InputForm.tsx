import { useState } from "react";
import { Send, SquarePen, StopCircle, Image as ImageIcon, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import type { ImageAttachment } from "@/lib/chat";

interface InputFormProps {
  onSubmit: (value: string, attachments?: ImageAttachment[]) => void;
  onCancel: () => void;
  isLoading: boolean;
  hasHistory: boolean;
}

export function InputForm({ onSubmit, onCancel, isLoading, hasHistory }: InputFormProps) {
  const [value, setValue] = useState("");
  const [attachments, setAttachments] = useState<{ filename: string; mime_type: string; data: string; previewUrl: string }[]>([]);

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    Array.from(files).forEach((file) => {
      if (!file.type.startsWith("image/")) return;
      const reader = new FileReader();
      reader.onload = (e) => {
        const result = e.target?.result as string;
        if (!result) return;
        const b64Data = result.split(",")[1];
        setAttachments((prev) => [
          ...prev,
          {
            filename: file.name,
            mime_type: file.type as string,
            data: b64Data,
            previewUrl: URL.createObjectURL(file),
          },
        ]);
      };
      reader.readAsDataURL(file);
    });
    event.target.value = "";
  };

  const removeAttachment = (index: number) => {
    setAttachments((prev) => prev.filter((_, i) => i !== index));
  };

  const submit = (event?: React.FormEvent) => {
    event?.preventDefault();
    if (!value.trim() && attachments.length === 0) return;
    const finalQuery = value.trim() || (attachments.length > 0 ? "Describe and analyze the attached image." : "");
    const finalAtts: ImageAttachment[] = attachments.map(({ filename, mime_type, data }) => ({ filename, mime_type, data }));
    onSubmit(finalQuery, finalAtts);
    setValue("");
    setAttachments([]);
  };

  return (
    <form onSubmit={submit} className="flex flex-col gap-2 p-3 pb-4">
      {attachments.length > 0 && (
        <div className="flex flex-wrap gap-2 px-2 pb-1">
          {attachments.map((att, idx) => (
            <div key={idx} className="relative flex items-center gap-2 rounded-lg bg-neutral-800 p-1.5 pr-6 border border-neutral-600 text-xs text-neutral-200">
              <img src={att.previewUrl} alt={att.filename} className="size-8 rounded object-cover" />
              <span className="truncate max-w-[120px]">{att.filename}</span>
              <button
                type="button"
                onClick={() => removeAttachment(idx)}
                className="absolute right-1 text-neutral-400 hover:text-neutral-100"
              >
                <X size={14} />
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="flex items-start rounded-3xl rounded-bl-sm bg-neutral-700 px-4 pt-3">
        <Textarea
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) submit();
          }}
          placeholder="Ask an enterprise question, attach an image, or profile a dataset…"
          className="border-0 text-neutral-100 placeholder:text-neutral-500"
          rows={1}
        />

        <label className="cursor-pointer text-neutral-400 hover:text-neutral-200 p-2">
          <ImageIcon size={20} />
          <input type="file" accept="image/jpeg,image/png,image/webp" multiple className="sr-only" onChange={handleImageUpload} disabled={isLoading} />
        </label>

        {isLoading ? (
          <Button type="button" variant="ghost" size="icon" className="text-red-400" onClick={onCancel}>
            <StopCircle />
          </Button>
        ) : (
          <Button type="submit" variant="ghost" size="icon" className="text-blue-400" disabled={!value.trim() && attachments.length === 0}>
            <Send />
          </Button>
        )}
      </div>

      {hasHistory && (
        <Button type="button" onClick={() => window.location.reload()} className="self-end">
          <SquarePen size={16} /> New conversation
        </Button>
      )}
    </form>
  );
}
