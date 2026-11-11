/**
 * Ocean Data Group emblem geometry: three crescent waves traced from
 * Only_Logo.png. Coordinates are in that image's pixel space.
 *
 * Each crescent is its outer circle, minus its cut circle, minus the next
 * crescent's outer circle grown by GAP (the white channel between waves).
 */

/** Framing used by the CRT boot-screen logo (includes stripe-shift headroom). */
export const VIEW = { x: 170, y: 50, w: 550, h: 560 } as const;

/** Tight box around the visible emblem, for clean (unstriped) renders. */
export const MARK_BOX = { x: 176, y: 86, w: 534, h: 504 } as const;

/** [cx, cy, r] for the outer and cut circle of each crescent, outer to inner. */
export const CRESCENTS = [
  { outer: [420, 330, 238], cut: [472, 345, 218] },
  { outer: [480, 360, 225], cut: [522, 343, 188] },
  { outer: [520, 355, 180], cut: [556, 318, 164] },
] as const;

/** Width of the white channel between neighbouring waves. */
export const GAP = 18;

/** Brand navy from the logo wordmark. */
export const ODG_NAVY = "#262467";
