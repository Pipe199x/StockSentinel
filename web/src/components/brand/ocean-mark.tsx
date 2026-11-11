import { useId } from "react";
import { cn } from "@/lib/utils";
import { CRESCENTS, GAP, MARK_BOX } from "./ocean-mark-geometry";

/**
 * SVG <defs> content that defines the emblem as `#${id}-shape` in currentColor.
 * Shared by <OceanMark> and the tiled watermark pattern.
 */
export function OceanMarkDefs({ id }: { id: string }) {
  return (
    <>
      {CRESCENTS.map(({ outer, cut }, i) => {
        const next = CRESCENTS[i + 1]?.outer;
        return (
          <mask
            key={i}
            id={`${id}-m${i}`}
            maskUnits="userSpaceOnUse"
            x={MARK_BOX.x - 20}
            y={MARK_BOX.y - 20}
            width={MARK_BOX.w + 40}
            height={MARK_BOX.h + 40}
          >
            <circle cx={outer[0]} cy={outer[1]} r={outer[2]} fill="white" />
            <circle cx={cut[0]} cy={cut[1]} r={cut[2]} fill="black" />
            {next && (
              <circle cx={next[0]} cy={next[1]} r={next[2] + GAP} fill="black" />
            )}
          </mask>
        );
      })}
      <g id={`${id}-shape`}>
        {CRESCENTS.map(({ outer }, i) => (
          <circle
            key={i}
            cx={outer[0]}
            cy={outer[1]}
            r={outer[2]}
            fill="currentColor"
            mask={`url(#${id}-m${i})`}
          />
        ))}
      </g>
    </>
  );
}

/** Clean single-color Ocean Data Group emblem. Inherits `color`. */
export function OceanMark({
  className,
  title,
}: {
  className?: string;
  /** Accessible name. Omit for a decorative (aria-hidden) mark. */
  title?: string;
}) {
  const id = `odg${useId().replace(/[^a-zA-Z0-9]/g, "")}`;
  return (
    <svg
      viewBox={`${MARK_BOX.x} ${MARK_BOX.y} ${MARK_BOX.w} ${MARK_BOX.h}`}
      className={cn("shrink-0", className)}
      {...(title ? { role: "img", "aria-label": title } : { "aria-hidden": true })}
    >
      <defs>
        <OceanMarkDefs id={id} />
      </defs>
      <use href={`#${id}-shape`} />
    </svg>
  );
}
