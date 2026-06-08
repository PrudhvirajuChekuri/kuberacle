"use client";

import { useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { FileText } from "lucide-react";

import type { Citation } from "@/lib/types";

const CARD_WIDTH = 290;
const EDGE_PAD = 8;
const GAP = 8;

/**
 * Source-preview card for an inline citation, rendered in a portal so it
 * escapes the answer scroll container's `overflow` clipping. Positions itself
 * above the anchor chip, centered and clamped to the viewport, flipping below
 * when there isn't room above.
 */
export function CitationPreview({
  anchorRect,
  citation,
}: {
  anchorRect: DOMRect;
  citation: Citation;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const { width, height } = el.getBoundingClientRect();
    const centerX = anchorRect.left + anchorRect.width / 2;
    const left = Math.min(
      Math.max(centerX - width / 2, EDGE_PAD),
      window.innerWidth - width - EDGE_PAD,
    );
    const above = anchorRect.top - GAP - height;
    const top = above >= EDGE_PAD ? above : anchorRect.bottom + GAP;
    setPos({ left, top });
  }, [anchorRect]);

  return createPortal(
    <div
      ref={ref}
      style={{
        left: pos?.left ?? 0,
        top: pos?.top ?? 0,
        width: CARD_WIDTH,
        visibility: pos ? "visible" : "hidden",
      }}
      className="pointer-events-none fixed z-50 flex flex-col gap-1.5 rounded-[11px] border border-border-2 bg-card p-3.5 text-left shadow-[var(--shadow)]"
    >
      <span className="flex items-start gap-1.5">
        <FileText className="mt-0.5 h-3.5 w-3.5 shrink-0 text-primary" />
        <span className="text-[13px] font-semibold leading-tight text-foreground">
          {citation.title || "Source"}
        </span>
      </span>
      {citation.snippet && (
        <span className="line-clamp-3 text-xs leading-relaxed text-text-2">{citation.snippet}</span>
      )}
      <span className="font-mono text-[11px] text-text-3">kubernetes.io</span>
    </div>,
    document.body,
  );
}
