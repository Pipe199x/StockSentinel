import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** clsx + tailwind-merge: a caller's className overrides a component's defaults. */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
