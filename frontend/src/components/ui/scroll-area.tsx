import * as React from "react";
import * as Primitive from "@radix-ui/react-scroll-area";
import { cn } from "@/lib/utils";
function ScrollArea({ className, children, ...props }: React.ComponentProps<typeof Primitive.Root>) { return <Primitive.Root className={cn("relative", className)} {...props}><Primitive.Viewport className="size-full rounded-[inherit]">{children}</Primitive.Viewport><Primitive.ScrollAreaScrollbar orientation="vertical" className="flex w-1.5 p-px"><Primitive.ScrollAreaThumb className="flex-1 rounded-full bg-neutral-600" /></Primitive.ScrollAreaScrollbar><Primitive.Corner /></Primitive.Root>; }
export { ScrollArea };
