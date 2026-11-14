import React, { useEffect } from "react";

/** Aceternity helper: calls `callback` on a mousedown/touchstart outside `ref`. */
export const useOutsideClick = (ref: React.RefObject<HTMLElement | null>, callback: (e: Event) => void) => {
  useEffect(() => {
    const listener = (event: Event) => {
      if (!ref.current || ref.current.contains(event.target as Node)) return;
      callback(event);
    };
    document.addEventListener("mousedown", listener);
    document.addEventListener("touchstart", listener);
    return () => {
      document.removeEventListener("mousedown", listener);
      document.removeEventListener("touchstart", listener);
    };
  }, [ref, callback]);
};
