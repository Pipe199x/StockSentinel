"use client";
import { cn } from "@/lib/utils";
import React, { useId } from "react";

export default function BackgroundGridWithDots() {
  return (
    <div className="relative flex min-h-screen w-full flex-col items-center justify-center overflow-hidden bg-white dark:bg-neutral-950">
      <GridWithDots />
      <Content />
    </div>
  );
}

const Content = () => {
  return (
    <div className="relative z-10 mx-auto max-w-3xl px-4">
      <h1 className="text-center text-3xl font-bold tracking-tight text-balance text-neutral-900 md:text-7xl dark:text-white">
        Build faster with our component library
      </h1>
      <p className="mx-auto mt-10 max-w-xl text-center text-xl text-balance text-neutral-600 md:mt-12 dark:text-neutral-400">
        Production-ready components designed to accelerate your development
        workflow.
      </p>
      <div className="mt-8 flex flex-col items-center justify-center gap-4 sm:flex-row md:mt-12">
        <button className="flex w-40 cursor-pointer items-center justify-center rounded-lg bg-neutral-900 px-4 py-2.5 text-sm font-medium text-white shadow-[0px_0px_10px_0px_rgba(255,255,255,0.2)_inset] ring ring-white/20 ring-offset-2 ring-offset-neutral-900 transition-all duration-200 ring-inset hover:shadow-[0px_0px_20px_0px_rgba(255,255,255,0.4)_inset] hover:ring-white/40 active:scale-98 dark:bg-white dark:text-black dark:shadow-[0px_0px_10px_0px_rgba(0,0,0,0.2)_inset] dark:ring-black/20 dark:ring-offset-white dark:hover:shadow-[0px_0px_20px_0px_rgba(0,0,0,0.3)_inset] dark:hover:ring-black/50">
          Get Started
        </button>
        <button className="flex w-40 cursor-pointer items-center justify-center rounded-lg border border-neutral-300 bg-transparent px-4 py-2.5 text-sm font-medium text-neutral-900 transition-all duration-200 hover:bg-neutral-100 active:scale-98 dark:border-neutral-700 dark:text-white dark:hover:bg-neutral-800">
          Learn More
        </button>
      </div>
    </div>
  );
};

interface GridWithDotsProps {
  gridSize?: number;
  dotSize?: number;
  lineColor?: string;
  dotColor?: string;
  className?: string;
}

const GridWithDots = ({
  gridSize = 80,
  dotSize = 3,
  lineColor,
  dotColor,
  className,
}: GridWithDotsProps) => {
  const patternId = useId();
  const maskId = useId();

  return (
    <div
      className={cn(
        "pointer-events-none absolute inset-0 h-full w-full",
        className,
      )}
    >
      <svg
        className="absolute inset-0 h-full w-full"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          <pattern
            id={patternId}
            width={gridSize}
            height={gridSize}
            patternUnits="userSpaceOnUse"
          >
            {/* Horizontal line through center */}
            <line
              x1="0"
              y1={gridSize / 2}
              x2={gridSize}
              y2={gridSize / 2}
              className={cn(
                "stroke-neutral-100 dark:stroke-neutral-900",
                lineColor,
              )}
              strokeWidth="1"
            />
            {/* Vertical line through center */}
            <line
              x1={gridSize / 2}
              y1="0"
              x2={gridSize / 2}
              y2={gridSize}
              className={cn(
                "stroke-neutral-100 dark:stroke-neutral-900",
                lineColor,
              )}
              strokeWidth="1"
            />
            {/* Dot at center intersection */}
            <circle
              cx={gridSize / 2}
              cy={gridSize / 2}
              r={dotSize}
              className={cn("fill-neutral-200 dark:fill-neutral-800", dotColor)}
            />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill={`url(#${patternId})`} />
      </svg>
    </div>
  );
};

export { GridWithDots };
