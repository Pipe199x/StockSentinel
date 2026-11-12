"use client";

import React, { useEffect, useRef, useMemo } from "react";
import { cn } from "@/lib/utils";

export default function LinesGradientShaderDemo() {
  return (
    <div className="relative h-screen w-full overflow-hidden">
      <LinesGradientShader
        className="absolute inset-0 bg-white dark:bg-neutral-950"
        bandCount={5}
        bandSpacing={40}
        bandThickness={100}
        waveAmplitude={0.2}
        speed={1}
      />
      <div className="relative z-10 mx-auto flex h-full max-w-2xl flex-col items-start justify-center px-8">
        <p className="mb-2 text-sm text-neutral-500 dark:text-neutral-400">
          New components and templates every month
        </p>
        <h1 className="mb-6 text-5xl font-medium tracking-tight text-neutral-900 dark:text-neutral-100">
          Build your next project in minutes, not hours.
        </h1>
        <p className="mb-8 text-xl text-neutral-600 dark:text-neutral-300">
          We are a team of developers who are passionate about creating web apps
          that make you feel like you&apos;re in the future.
        </p>
        <div className="flex flex-col gap-4 md:flex-row">
          <button className="rounded-full bg-indigo-600 px-6 py-3 text-sm font-medium text-white transition-colors hover:bg-indigo-700 dark:bg-indigo-500 dark:hover:bg-indigo-600">
            Get started →
          </button>
          <button className="rounded-full border border-neutral-300 bg-white px-6 py-3 text-sm font-medium text-neutral-700 transition-colors hover:bg-neutral-50 dark:border-neutral-600 dark:bg-neutral-800 dark:text-neutral-200 dark:hover:bg-neutral-700">
            <span className="mr-2">G</span> Sign up with Google
          </button>
        </div>
      </div>
    </div>
  );
}

interface LinesGradientShaderProps {
  /** CSS class name for the container (use Tailwind for background color) */
  className?: string;
  /** Animation speed multiplier (lower = slower) */
  speed?: number;
  /** Number of bands in the spiral */
  bandCount?: number;
  /** Spacing between band centers in pixels */
  bandSpacing?: number;
  /** Thickness of each band in pixels */
  bandThickness?: number;
  /** Wave amplitude as a fraction of container height (0-1) */
  waveAmplitude?: number;
  /** Custom color stops for the gradient (array of RGB tuples) */
  colors?: [number, number, number][];
  /** Target FPS (default: 30 for better performance) */
  targetFps?: number;
}

const DEFAULT_COLORS: [number, number, number][] = [
  [139, 92, 246], // violet-500
  [168, 85, 247], // purple-500
  [217, 70, 239], // fuchsia-500
  [236, 72, 153], // pink-500
  [244, 63, 94], // rose-500
  [249, 115, 22], // orange-500
  [251, 191, 36], // amber-400
  [254, 240, 138], // yellow-200
];

export const LinesGradientShader: React.FC<LinesGradientShaderProps> = ({
  className,
  speed = 1,
  bandCount = 12,
  bandSpacing = 25,
  bandThickness = 60,
  waveAmplitude = 0.15,
  colors,
  targetFps = 30,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef<number>(0);
  const startTimeRef = useRef<number>(0);
  const lastFrameTimeRef = useRef<number>(0);
  const isVisibleRef = useRef<boolean>(true);

  const colorStops = useMemo(() => colors || DEFAULT_COLORS, [colors]);

  const precomputedColors = useMemo(() => {
    const interpolateColor = (
      color1: [number, number, number],
      color2: [number, number, number],
      t: number,
    ): string => {
      const r = Math.round(color1[0] + (color2[0] - color1[0]) * t);
      const g = Math.round(color1[1] + (color2[1] - color1[1]) * t);
      const b = Math.round(color1[2] + (color2[2] - color1[2]) * t);
      return `rgb(${r},${g},${b})`;
    };

    const getColorAtPosition = (t: number): string => {
      const clampedT = Math.max(0, Math.min(1, t));
      const scaledT = clampedT * (colorStops.length - 1);
      const index = Math.floor(scaledT);
      const fraction = scaledT - index;
      const color1 = colorStops[Math.min(index, colorStops.length - 1)];
      const color2 = colorStops[Math.min(index + 1, colorStops.length - 1)];
      return interpolateColor(color1, color2, fraction);
    };

    const bandColors: { start: string; end: string }[] = [];
    for (let i = 0; i < bandCount; i++) {
      const progress = i / (bandCount - 1);
      bandColors.push({
        start: getColorAtPosition(progress - 0.02),
        end: getColorAtPosition(progress + 0.08),
      });
    }
    return bandColors;
  }, [colorStops, bandCount]);

  useEffect(() => {
    // Reduced motion: draw a single still frame instead of nothing (ODG change)
    const prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;

    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext("2d", {
      alpha: true,
      desynchronized: true,
    });
    if (!ctx) return;

    let stopped = false;
    const dpr = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
    const frameInterval = 1000 / targetFps;
    const steps = 40;
    const highlightSteps = 30;

    let cachedWidth = 0;
    let cachedHeight = 0;

    const resize = () => {
      const { width, height } = container.getBoundingClientRect();
      cachedWidth = width;
      cachedHeight = height;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${Math.floor(width)}px`;
      canvas.style.height = `${Math.floor(height)}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (prefersReducedMotion && stopped) {
        // a resize clears the canvas: redraw the still frame
        stopped = false;
        lastFrameTimeRef.current = 0;
        rafRef.current = requestAnimationFrame(draw);
      }
    };

    const ro = new ResizeObserver(resize);
    ro.observe(container);
    resize();

    const io = new IntersectionObserver(
      (entries) => {
        isVisibleRef.current = entries[0]?.isIntersecting ?? true;
      },
      { threshold: 0 },
    );
    io.observe(container);

    const handleVisibilityChange = () => {
      if (!document.hidden && isVisibleRef.current) {
        startTimeRef.current = 0;
        lastFrameTimeRef.current = 0;
      }
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);

    const bottomPoints: { x: number; y: number }[] = new Array(steps + 1);
    const topPoints: { x: number; y: number }[] = new Array(steps + 1);
    for (let j = 0; j <= steps; j++) {
      bottomPoints[j] = { x: 0, y: 0 };
      topPoints[j] = { x: 0, y: 0 };
    }

    const draw = (timestamp: number) => {
      if (stopped) return;

      rafRef.current = requestAnimationFrame(draw);

      if (!isVisibleRef.current || document.hidden) {
        return;
      }

      const elapsed = timestamp - lastFrameTimeRef.current;
      if (elapsed < frameInterval) {
        return;
      }
      lastFrameTimeRef.current = timestamp - (elapsed % frameInterval);

      if (startTimeRef.current === 0) {
        startTimeRef.current = timestamp;
      }

      const time = (timestamp - startTimeRef.current) * 0.001 * speed;
      const width = cachedWidth;
      const height = cachedHeight;

      ctx.clearRect(0, 0, width, height);

      const baseAmplitude = height * waveAmplitude;
      const halfBandCount = bandCount / 2;
      const widthRange = width + 400;

      for (let i = bandCount - 1; i >= 0; i--) {
        const { start: colorStart, end: colorEnd } = precomputedColors[i];

        const gradient = ctx.createLinearGradient(
          width * 0.3,
          0,
          width,
          height,
        );
        gradient.addColorStop(0, colorStart);
        gradient.addColorStop(1, colorEnd);

        ctx.fillStyle = gradient;
        ctx.beginPath();

        const phase1 = time * 0.12 + i * 0.15;
        const phase2 = time * 0.08 + i * 0.1;
        const phase3 = time * 0.05 + i * 0.08;
        const bandOffset = (i - halfBandCount) * bandSpacing;

        ctx.moveTo(-100, height + 200);

        for (let j = 0; j <= steps; j++) {
          const t = j / steps;
          const x = -100 + widthRange * t;
          const baseY = height * 1.4 - t * height * 1.2 + bandOffset;

          const wave1 = Math.sin(t * 2.5 + phase1) * baseAmplitude;
          const wave2 = Math.sin(t * 1.5 + phase2) * baseAmplitude * 0.4;
          const wave3 = Math.sin(t * 4 + phase3) * baseAmplitude * 0.15;
          const waveOffset = wave1 + wave2 + wave3;

          const thickness = bandThickness + 4 * Math.sin(t * 2 + phase1 * 0.3);
          const halfThickness = thickness / 2;

          bottomPoints[j].x = x;
          bottomPoints[j].y = baseY + waveOffset + halfThickness;
          topPoints[j].x = x;
          topPoints[j].y = baseY + waveOffset - halfThickness;
        }

        for (let j = 0; j <= steps; j++) {
          ctx.lineTo(bottomPoints[j].x, bottomPoints[j].y);
        }

        ctx.lineTo(width + 200, -100);

        for (let j = steps; j >= 0; j--) {
          ctx.lineTo(topPoints[j].x, topPoints[j].y);
        }

        ctx.lineTo(-100, height + 200);
        ctx.closePath();
        ctx.fill();
      }

      ctx.globalCompositeOperation = "overlay";
      ctx.globalAlpha = 0.25;
      ctx.strokeStyle = "rgba(255,255,255,0.5)";
      ctx.lineWidth = 1;

      for (let i = 0; i < bandCount; i++) {
        const phase1 = time * 0.12 + i * 0.15;
        const phase2 = time * 0.08 + i * 0.1;
        const phase3 = time * 0.05 + i * 0.08;
        const bandOffset = (i - halfBandCount) * bandSpacing;

        ctx.beginPath();

        for (let j = 0; j <= highlightSteps; j++) {
          const t = j / highlightSteps;
          const x = -100 + widthRange * t;
          const baseY = height * 1.4 - t * height * 1.2 + bandOffset;

          const wave1 = Math.sin(t * 2.5 + phase1) * baseAmplitude;
          const wave2 = Math.sin(t * 1.5 + phase2) * baseAmplitude * 0.4;
          const wave3 = Math.sin(t * 4 + phase3) * baseAmplitude * 0.15;
          const y = baseY + wave1 + wave2 + wave3;

          if (j === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }

        ctx.stroke();
      }

      ctx.globalCompositeOperation = "source-over";
      ctx.globalAlpha = 1;

      if (prefersReducedMotion) {
        stopped = true;
        cancelAnimationFrame(rafRef.current);
      }
    };

    rafRef.current = requestAnimationFrame(draw);

    return () => {
      stopped = true;
      cancelAnimationFrame(rafRef.current);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      ro.disconnect();
      io.disconnect();
    };
  }, [
    speed,
    bandCount,
    bandSpacing,
    bandThickness,
    waveAmplitude,
    precomputedColors,
    targetFps,
  ]);

  return (
    <div
      ref={containerRef}
      className={cn("relative overflow-hidden", className)}
    >
      <canvas
        ref={canvasRef}
        className="absolute inset-0 h-full w-full"
        style={{ display: "block" }}
      />
    </div>
  );
};
