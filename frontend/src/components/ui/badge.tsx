import * as React from "react";
import { cn } from "@/lib/utils";
function Badge({ className, ...props }: React.ComponentProps<"span">) { return <span className={cn("inline-flex rounded bg-neutral-700 px-1.5 py-0.5", className)} {...props} />; }
export { Badge };
