import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva("inline-flex items-center justify-center gap-2 rounded-md text-sm font-medium transition-all disabled:pointer-events-none disabled:opacity-50", { variants: { variant: { default: "bg-neutral-700 text-neutral-100 hover:bg-neutral-600", destructive: "bg-red-700 text-white hover:bg-red-600", ghost: "hover:bg-neutral-700" }, size: { default: "h-9 px-4 py-2", icon: "size-9" } }, defaultVariants: { variant: "default", size: "default" } });
function Button({ className, variant, size, asChild = false, ...props }: React.ComponentProps<"button"> & VariantProps<typeof buttonVariants> & { asChild?: boolean }) { const Comp = asChild ? Slot : "button"; return <Comp className={cn(buttonVariants({ variant, size, className }))} {...props} />; }
export { Button, buttonVariants };
