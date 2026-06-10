import { ExternalLink, FileText } from "lucide-react";

import { cn } from "@/lib/utils";
import type { Citation } from "@/lib/types";

/** Title-case a URL path segment (e.g. "workload-resources" -> "Workload Resources"). */
function titleCase(segment: string): string {
  return segment.replace(/[-_]/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Build a breadcrumb from a docs URL path (e.g. "Concepts › Workloads › Controllers"). */
function breadcrumb(url: string): string {
  try {
    const path = new URL(url).pathname.replace(/^\/|\/$/g, "");
    const segments = path.split("/").filter((s) => s && s !== "docs");
    return segments.map(titleCase).join(" › ");
  } catch {
    return "";
  }
}

/** Fallback title from the last path segment when the citation has no title. */
function fallbackTitle(url: string): string {
  try {
    const segments = new URL(url).pathname.replace(/^\/|\/$/g, "").split("/").filter(Boolean);
    return titleCase(segments[segments.length - 1] ?? "Source");
  } catch {
    return "Source";
  }
}

interface CitationCardProps {
  citation: Citation;
  /** DOM id used as the scroll target for inline `[n]` references. */
  anchorId: string;
  /** Highlights the card when its inline `[n]` marker was clicked. */
  active?: boolean;
}

export function CitationCard({ citation, anchorId, active }: CitationCardProps) {
  const title = citation.title || fallbackTitle(citation.source_url);
  const crumb = breadcrumb(citation.source_url);

  return (
    <a
      id={anchorId}
      href={citation.source_url}
      target="_blank"
      rel="noopener noreferrer"
      className={cn(
        "group block min-w-0 rounded-[11px] border border-border bg-card p-4 transition hover:-translate-y-px hover:border-brand-line",
        active && "border-primary shadow-[0_0_0_1px_var(--primary)]",
      )}
    >
      <div className="flex items-start gap-3">
        <span className="grid h-[22px] w-[22px] shrink-0 place-items-center rounded-[6px] bg-brand-soft text-primary">
          <FileText className="h-3.5 w-3.5" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-start justify-between gap-2">
            <span className="min-w-0 flex-1 break-words text-[13.5px] font-semibold leading-tight text-foreground">
              {title}
            </span>
            <ExternalLink className="mt-0.5 h-3.5 w-3.5 shrink-0 text-text-3 transition-colors group-hover:text-primary" />
          </div>
          {crumb && (
            <div className="mt-1 truncate font-mono text-[11px] text-text-3">{crumb}</div>
          )}
          {citation.snippet && (
            <p className="mt-2 line-clamp-2 text-[12.5px] leading-relaxed text-text-2">
              {citation.snippet}
            </p>
          )}
          <div className="mt-2.5 flex items-center gap-2.5 font-mono text-[11px] text-text-3">
            <span>kubernetes.io</span>
          </div>
        </div>
      </div>
    </a>
  );
}
