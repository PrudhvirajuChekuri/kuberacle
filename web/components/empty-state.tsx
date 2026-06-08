import { ArrowRight } from "lucide-react";

import { ChatInput } from "@/components/chat-input";
import { Cube } from "@/components/cube-icon";
import { K8S_VERSION } from "@/lib/constants";

const SUGGESTIONS = [
  { tag: "concept", label: "What is a Pod?", q: "What is a Pod?" },
  {
    tag: "compare",
    label: "How does a Deployment differ from a StatefulSet?",
    q: "How does a Deployment differ from a StatefulSet?",
  },
  {
    tag: "task",
    label: "How do I expose a Service outside the cluster?",
    q: "How do I expose a Service outside the cluster?",
  },
];

/** Landing hero shown before the first question: mark, prompt, suggestions. */
export function EmptyState({ onSend }: { onSend: (value: string) => void }) {
  return (
    <div className="mx-auto flex max-w-[660px] flex-col items-center px-6 pb-16 pt-[8vh] text-center">
      <div className="mb-7 grid h-[68px] w-[68px] place-items-center rounded-[18px] bg-[linear-gradient(150deg,var(--brand),color-mix(in_srgb,var(--brand)_65%,#000))] text-white shadow-[0_18px_48px_-18px_var(--brand-line)]">
        <Cube className="h-8 w-8" strokeWidth={1.5} />
      </div>
      <h1 className="mb-3 text-[34px] font-bold tracking-tight">Ask the Kubernetes docs</h1>
      <p className="mb-7 text-base text-text-2">
        Answers are grounded in{" "}
        <strong className="font-semibold text-foreground">kubernetes.io</strong> and cite the exact
        pages they came from.
      </p>
      <div className="mb-7 w-full">
        <ChatInput
          big
          onSend={onSend}
          onStop={() => {}}
          isStreaming={false}
          placeholder="e.g. How do I configure a liveness probe?"
        />
      </div>
      <div className="flex w-full flex-col gap-2.5">
        {SUGGESTIONS.map((s) => (
          <button
            key={s.q}
            onClick={() => onSend(s.q)}
            className="group flex w-full items-center gap-3 rounded-[11px] border border-border bg-card px-4 py-3.5 text-left transition hover:-translate-y-px hover:border-brand-line"
          >
            <span className="shrink-0 rounded-md bg-brand-soft px-2 py-0.5 font-mono text-[10.5px] uppercase tracking-wide text-primary">
              {s.tag}
            </span>
            <span className="flex-1 text-[14.5px] font-medium">{s.label}</span>
            <ArrowRight className="h-[15px] w-[15px] text-text-3 transition-colors group-hover:text-primary" />
          </button>
        ))}
      </div>
      <div className="mt-7 flex items-center gap-3 font-mono text-xs text-text-3">
        <span>Kubernetes {K8S_VERSION}</span>
        <span className="h-[3px] w-[3px] rounded-full bg-text-3" />
        <span>citations on every answer</span>
      </div>
    </div>
  );
}
