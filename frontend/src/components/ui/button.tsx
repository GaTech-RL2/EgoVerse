import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex h-9 items-center justify-center gap-2 rounded-md border px-3 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/60 disabled:pointer-events-none disabled:opacity-45",
  {
    variants: {
      variant: {
        default: "border-cyan-300/30 bg-cyan-300 text-slate-950 hover:bg-cyan-200",
        outline: "border-white/12 bg-white/[0.03] text-slate-200 hover:bg-white/[0.07]",
        ghost: "border-transparent bg-transparent text-slate-400 hover:bg-white/[0.05] hover:text-slate-100",
      },
    },
    defaultVariants: { variant: "default" },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export function Button({ className, variant, ...props }: ButtonProps) {
  return <button className={cn(buttonVariants({ variant }), className)} {...props} />;
}
