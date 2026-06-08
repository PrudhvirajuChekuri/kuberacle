"use client";

import { useState, type ReactNode } from "react";
import { Check, Copy } from "lucide-react";

/** Color a scalar YAML value: numbers/bools/null get the accent, else plain. */
function colorVal(v: string): ReactNode {
  if (!v) return v;
  const t = v.trim();
  if (/^-?\d+(\.\d+)?$/.test(t) || /^(true|false|null)$/.test(t)) {
    return <span className="text-[#e0af68]">{v}</span>;
  }
  return v;
}

/** Render a `key: value` pair with key/punctuation/value coloring. */
function renderKv(rest: string): ReactNode {
  const m = rest.match(/^([A-Za-z0-9_.-]+)(:)(.*)$/);
  if (m) {
    return (
      <>
        <span className="text-[#7aa2f7]">{m[1]}</span>
        <span className="text-[#566074]">:</span>
        <span className="text-[#c2cde0]">{colorVal(m[3])}</span>
      </>
    );
  }
  return <span className="text-[#c2cde0]">{colorVal(rest)}</span>;
}

/** Lightweight line-based YAML highlighter (keys, values, comments, lists). */
function highlightYaml(code: string): ReactNode[] {
  return code.split("\n").map((line, i) => {
    const indent = line.match(/^(\s*)/)?.[1] ?? "";
    let rest = line.slice(indent.length);
    const segs: ReactNode[] = [<span key="i">{indent}</span>];
    if (rest.startsWith("#")) {
      segs.push(
        <span key="c" className="italic text-[#5b6675]">
          {rest}
        </span>,
      );
    } else if (rest.startsWith("- ")) {
      segs.push(
        <span key="d" className="text-[#566074]">
          {"- "}
        </span>,
      );
      rest = rest.slice(2);
      segs.push(<span key="kv">{renderKv(rest)}</span>);
    } else {
      segs.push(<span key="kv">{renderKv(rest)}</span>);
    }
    return (
      <div key={i} className="whitespace-pre">
        {segs}
      </div>
    );
  });
}

const YAML_LANGS = new Set(["yaml", "yml"]);

/** A fenced code block with a header (language + copy) and optional YAML coloring. */
export function CodeBlock({ code, lang }: { code: string; lang: string }) {
  const [copied, setCopied] = useState(false);

  const copy = () => {
    navigator.clipboard?.writeText(code).catch(() => {});
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  };

  return (
    <div className="my-4 overflow-hidden rounded-[11px] border border-code-border bg-code-bg">
      <div className="flex items-center justify-between border-b border-code-border bg-[color-mix(in_srgb,var(--code-bg)_70%,#1a2230)] px-3 py-2.5">
        <div className="flex items-center gap-2.5">
          <span className="h-2.5 w-2.5 rounded-full bg-primary shadow-[0_0_0_3px_color-mix(in_srgb,var(--brand)_22%,transparent)]" />
          <span className="font-mono text-[10px] uppercase tracking-wider text-[#6b7684]">
            {lang || "text"}
          </span>
        </div>
        <button
          onClick={copy}
          className="flex items-center gap-1.5 rounded-md border border-[#2a3240] px-2.5 py-1 font-mono text-xs text-[#9aa6b5] transition hover:border-[#3a4456] hover:text-[#e7ebf0]"
        >
          {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="overflow-x-auto px-4 py-3.5 font-mono text-[12.8px] leading-[1.7] text-[#aeb9cc]">
        <code>{YAML_LANGS.has(lang) ? highlightYaml(code) : code}</code>
      </pre>
    </div>
  );
}
