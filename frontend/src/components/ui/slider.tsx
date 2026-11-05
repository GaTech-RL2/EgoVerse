"use client";

import * as SliderPrimitive from "@radix-ui/react-slider";
import * as React from "react";

import { cn } from "@/lib/utils";

export function Slider({ className, ...props }: React.ComponentProps<typeof SliderPrimitive.Root>) {
  return (
    <SliderPrimitive.Root
      className={cn("precision-slider", className)}
      {...props}
    >
      <SliderPrimitive.Track className="precision-slider-track">
        <SliderPrimitive.Range className="precision-slider-range" />
      </SliderPrimitive.Track>
      <SliderPrimitive.Thumb className="precision-slider-thumb" />
    </SliderPrimitive.Root>
  );
}
