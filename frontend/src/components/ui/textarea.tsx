import * as React from "react";
import { cn } from "@/lib/utils";
function Textarea({ className, ...props }: React.ComponentProps<"textarea">) { return <textarea className={cn("min-h-16 w-full resize-none rounded-md border border-neutral-600 bg-transparent px-3 py-2 text-base outline-none focus:border-neutral-400 disabled:cursor-not-allowed disabled:opacity-50", className)} {...props} />; }
export { Textarea };
