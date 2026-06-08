"use client";

import { useState, type MouseEvent, type ReactNode } from "react";
import { AlertTriangle, FileText, Layers, Search } from "lucide-react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import { CitationCard } from "@/components/citation-card";
import { CodeBlock } from "@/components/code-block";
import { CopyButton } from "@/components/copy-button";
import { Cube } from "@/components/cube-icon";
import { cn } from "@/lib/utils";
import type { ChatMessage, Citation } from "@/lib/types";

const CITATION_MARKER = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

/** Turn `[n]` / `[n, m]` markers into in-page links to their citation cards. */
function linkifyCitations(text: string, messageId: string): string {
  return text.replace(CITATION_MARKER, (_match, group: string) =>
    group
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean)
      .map((n) => `[\\[${n}\\]](#cite-${messageId}-${n})`)
      .join(""),
  );
}

/** Markdown renderers matching the answer-body design system. */
const baseComponents: Components = {
  p: ({ children }) => <p className="mb-3.5 text-pretty">{children}</p>,
  strong: ({ children }) => <strong className="font-semibold text-foreground">{children}</strong>,
  h1: ({ children }) => <h3 className="mb-3 mt-5 text-base font-semibold tracking-tight">{children}</h3>,
  h2: ({ children }) => <h3 className="mb-3 mt-5 text-base font-semibold tracking-tight">{children}</h3>,
  h3: ({ children }) => <h3 className="mb-3 mt-5 text-base font-semibold tracking-tight">{children}</h3>,
  ul: ({ children }) => <ul className="mb-4 flex list-none flex-col gap-2.5 pl-0">{children}</ul>,
  ol: ({ children }) => <ol className="mb-4 flex list-decimal flex-col gap-2.5 pl-5">{children}</ol>,
  li: ({ children }) => (
    <li className="relative pl-5 text-pretty before:absolute before:left-1 before:top-[0.62em] before:h-[5px] before:w-[5px] before:rounded-full before:bg-primary before:content-['']">
      {children}
    </li>
  ),
  // The fenced-block container is provided by CodeBlock, so unwrap <pre>.
  pre: ({ children }) => <>{children}</>,
  code: ({ className, children }) => {
    if (className && className.includes("language-")) {
      const lang = /language-(\w+)/.exec(className)?.[1] ?? "";
      const text = (Array.isArray(children) ? children.join("") : String(children)).replace(
        /\n$/,
        "",
      );
      return <CodeBlock code={text} lang={lang} />;
    }
    return (
      <code className="rounded border border-border bg-surface-2 px-1.5 py-0.5 font-mono text-[0.86em] text-brand-dim dark:text-[#9ec1ff]">
        {children}
      </code>
    );
  },
};

/** Inline `[n]` citation chip with a source-preview card on hover. */
function CitationChip({
  href,
  citation,
  onSelect,
  children,
}: {
  href: string;
  citation?: Citation;
  onSelect: () => void;
  children: ReactNode;
}) {
  const handleClick = (e: MouseEvent) => {
    e.preventDefault();
    onSelect();
  };

  const chip = (
    <a
      href={href}
      onClick={handleClick}
      className="mx-0.5 inline-flex h-4 min-w-[17px] items-center justify-center rounded-md bg-brand-soft px-1 align-super font-mono text-[10.5px] font-semibold text-primary no-underline transition-colors hover:bg-primary hover:text-white"
    >
      {children}
    </a>
  );
  if (!citation) return chip;
  return (
    <span className="group relative inline-block">
      {chip}
      <span className="pointer-events-none invisible absolute bottom-[calc(100%+8px)] left-1/2 z-40 flex w-[290px] -translate-x-1/2 flex-col gap-1.5 rounded-[11px] border border-border-2 bg-card p-3.5 text-left opacity-0 shadow-[var(--shadow)] transition-opacity group-hover:visible group-hover:opacity-100">
        <span className="flex items-start gap-1.5">
          <FileText className="mt-0.5 h-3.5 w-3.5 shrink-0 text-primary" />
          <span className="text-[13px] font-semibold leading-tight text-foreground">
            {citation.title || "Source"}
          </span>
        </span>
        {citation.snippet && (
          <span className="line-clamp-3 text-xs leading-relaxed text-text-2">
            {citation.snippet}
          </span>
        )}
        <span className="font-mono text-[11px] text-text-3">kubernetes.io</span>
      </span>
    </span>
  );
}

function ThinkingDots() {
  return (
    <div className="flex items-center gap-2 py-1 font-mono text-sm text-text-3">
      <span className="flex gap-1">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="h-1.5 w-1.5 animate-bounce rounded-full bg-primary"
            style={{ animationDelay: `${i * 0.15}s` }}
          />
        ))}
      </span>
      searching kubernetes.io docs…
    </div>
  );
}

export function Message({ message }: { message: ChatMessage }) {
  const [activeCite, setActiveCite] = useState<number | null>(null);

  if (message.role === "user") {
    return (
      <div className="flex items-center gap-3 rounded-xl border border-border bg-surface-2 px-4 py-3">
        <Search className="h-[18px] w-[18px] shrink-0 text-text-3" />
        <h2 className="whitespace-pre-wrap text-[17px] font-semibold tracking-tight">
          {message.content}
        </h2>
      </div>
    );
  }

  const showTyping = message.pending && message.content === "";
  const isAbstention = message.content.trim().startsWith("INSUFFICIENT_EVIDENCE");

  const selectCite = (n: number) => {
    setActiveCite(n);
    document
      .getElementById(`cite-${message.id}-${n}`)
      ?.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  const citationByIndex = new Map<number, Citation>(
    (message.citations ?? []).map((citation) => [citation.index, citation]),
  );
  const components: Components = {
    ...baseComponents,
    a: ({ href, children, ...props }) => {
      if (href?.startsWith("#cite-")) {
        const n = Number(href.split("-").pop());
        return (
          <CitationChip href={href} citation={citationByIndex.get(n)} onSelect={() => selectCite(n)}>
            {children}
          </CitationChip>
        );
      }
      return (
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className="text-primary underline-offset-2 hover:underline"
          {...props}
        >
          {children}
        </a>
      );
    },
  };

  return (
    <div className="flex items-start gap-3">
      <div className="mt-0.5 grid h-[30px] w-[30px] shrink-0 place-items-center rounded-lg bg-primary text-white shadow-[0_4px_12px_-5px_var(--brand-line)]">
        <Cube className="h-4 w-4" />
      </div>
      <div className="min-w-0 flex-1">
        {showTyping ? (
          <ThinkingDots />
        ) : isAbstention ? (
          <p className="text-[15.5px] text-text-2">
            I couldn&apos;t find an answer to this in the Kubernetes documentation. Try rephrasing,
            or ask about a different topic.
          </p>
        ) : (
          <div
            className={cn(
              "text-[15.5px] leading-[1.68] text-foreground break-words",
              message.error && "text-destructive",
            )}
          >
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
              {linkifyCitations(message.content, message.id)}
            </ReactMarkdown>
            {message.pending && (
              <span className="ml-0.5 inline-block h-[17px] w-2 translate-y-[3px] animate-pulse rounded-[1px] bg-primary align-text-bottom" />
            )}
          </div>
        )}

        {message.insufficientEvidence && !isAbstention && (
          <div className="mt-4 flex items-center gap-2 rounded-lg border border-amber-500/20 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-400">
            <AlertTriangle className="h-4 w-4 shrink-0" />
            Couldn&apos;t verify sources for this answer — treat it as ungrounded.
          </div>
        )}

        {message.citations && message.citations.length > 0 && (
          <div className="mt-6">
            <div className="mb-3 flex items-center gap-1.5 font-mono text-xs uppercase tracking-wide text-text-3">
              <Layers className="h-3.5 w-3.5" /> {message.citations.length} sources from
              kubernetes.io
            </div>
            <div className="grid gap-2.5 sm:grid-cols-2">
              {[...message.citations]
                .sort((a, b) => a.index - b.index)
                .map((citation) => (
                  <CitationCard
                    key={citation.chunk_id}
                    citation={citation}
                    anchorId={`cite-${message.id}-${citation.index}`}
                    active={activeCite === citation.index}
                  />
                ))}
            </div>
          </div>
        )}

        {!message.pending && !isAbstention && message.content && (
          <div className="mt-5 flex justify-end">
            <CopyButton text={message.content} label="Copy response" />
          </div>
        )}
      </div>
    </div>
  );
}
