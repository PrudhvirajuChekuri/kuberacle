import { ExternalLink, FileText } from "lucide-react";

import type { Citation } from "@/lib/types";

/** Render a human-readable label for a source URL (path + hash). */
function urlLabel(url: string): string {
  try {
    const parsed = new URL(url);
    return `${parsed.pathname}${parsed.hash}`.replace(/\/$/, "");
  } catch {
    return url;
  }
}

interface CitationCardProps {
  /** 1-based citation index, matching the `[n]` markers in the answer. */
  index: number;
  citation: Citation;
  /** DOM id used as the scroll target for inline `[n]` references. */
  anchorId: string;
}

export function CitationCard({ index, citation, anchorId }: CitationCardProps) {
  return (
    <a
      id={anchorId}
      href={citation.source_url}
      target="_blank"
      rel="noopener noreferrer"
      className="group flex items-start gap-3 rounded-[11px] border border-border bg-card p-3.5 transition target:border-primary target:ring-1 target:ring-primary hover:-translate-y-px hover:border-brand-line"
    >
      <span className="grid h-[22px] w-[22px] shrink-0 place-items-center rounded-md bg-brand-soft font-mono text-xs font-semibold text-primary">
        {index}
      </span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-[13.5px] font-semibold leading-tight text-foreground">
          {urlLabel(citation.source_url)}
        </span>
        <span className="mt-2 flex items-center gap-2.5 font-mono text-[11px] text-text-3">
          <span className="flex items-center gap-1.5">
            <FileText className="h-3 w-3" /> kubernetes.io
          </span>
          <span className="rounded border border-border px-1.5 py-px">
            relevance {(citation.score * 100).toFixed(0)}%
          </span>
          <ExternalLink className="ml-auto h-3 w-3 transition-colors group-hover:text-primary" />
        </span>
      </span>
    </a>
  );
}
