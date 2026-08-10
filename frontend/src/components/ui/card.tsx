import * as React from "react";
import { cn } from "@/lib/utils";
function Card({ className, ...props }: React.ComponentProps<"div">) { return <div className={cn("rounded-lg", className)} {...props} />; }
function CardHeader({ className, ...props }: React.ComponentProps<"div">) { return <div className={cn("p-4 pb-2", className)} {...props} />; }
function CardContent({ className, ...props }: React.ComponentProps<"div">) { return <div className={cn("p-4 pt-2", className)} {...props} />; }
function CardDescription({ className, ...props }: React.ComponentProps<"div">) { return <div className={cn("text-sm", className)} {...props} />; }
export { Card, CardHeader, CardContent, CardDescription };
