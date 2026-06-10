"use client";

import { useState } from "react";
import { ArrowUp, Search, Square } from "lucide-react";

import { cn } from "@/lib/utils";

interface ChatInputProps {
  onSend: (value: string) => void;
  onStop: () => void;
  isStreaming: boolean;
  big?: boolean;
  placeholder?: string;
}

export function ChatInput({ onSend, onStop, isStreaming, big, placeholder }: ChatInputProps) {
  const [value, setValue] = useState("");

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || isStreaming) return;
    onSend(trimmed);
    setValue("");
  };

  return (
    <div
      className={cn(
        "group flex items-center gap-2.5 border border-border-2 bg-card shadow-[var(--shadow)] transition-colors focus-within:border-primary",
        big ? "rounded-[15px] px-4 py-2.5" : "rounded-[13px] px-3.5 py-2",
      )}
    >
      <Search className="h-[18px] w-[18px] shrink-0 text-text-3 transition-colors group-focus-within:text-primary" />
      <textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
        placeholder={placeholder ?? "Ask about Kubernetes…"}
        rows={1}
        className={cn(
          "max-h-40 min-h-0 flex-1 resize-none border-0 bg-transparent text-foreground outline-none placeholder:text-text-3",
          big ? "py-1.5 text-[16.5px]" : "py-1 text-[16px]",
        )}
      />
      {isStreaming ? (
        <button
          onClick={onStop}
          aria-label="Stop"
          className="grid h-9 w-9 shrink-0 place-items-center rounded-[10px] bg-secondary text-foreground transition hover:opacity-90"
        >
          <Square className="h-4 w-4" />
        </button>
      ) : (
        <button
          onClick={submit}
          disabled={!value.trim()}
          aria-label="Send"
          className={cn(
            "grid shrink-0 place-items-center rounded-[10px] bg-primary text-white transition hover:-translate-y-px disabled:translate-y-0 disabled:opacity-40",
            big ? "h-[42px] w-[42px]" : "h-[38px] w-[38px]",
          )}
        >
          <ArrowUp className="h-4 w-4" />
        </button>
      )}
    </div>
  );
}
