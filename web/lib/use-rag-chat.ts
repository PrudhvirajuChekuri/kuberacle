"use client";

import { useCallback, useRef, useState } from "react";

import { parseSSE } from "@/lib/sse";
import type { ChatMessage, FinalEventData } from "@/lib/types";

let counter = 0;
const nextId = () => `${Date.now()}-${counter++}`;

/**
 * Chat state hook backed by the streaming `/api/query` SSE endpoint.
 *
 * Appends a user message and a placeholder assistant message, then fills the
 * assistant message from `token` deltas and finalizes it on the `final` event.
 */
export function useRagChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const send = useCallback(
    async (question: string) => {
      const trimmed = question.trim();
      if (!trimmed || isStreaming) return;

      const assistantId = nextId();
      setMessages((prev) => [
        ...prev,
        { id: nextId(), role: "user", content: trimmed },
        { id: assistantId, role: "assistant", content: "", pending: true },
      ]);
      setIsStreaming(true);

      const patch = (update: Partial<ChatMessage>) =>
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, ...update } : m)),
        );

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const res = await fetch("/api/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: trimmed }),
          signal: controller.signal,
        });

        if (!res.ok || !res.body) {
          patch({ pending: false, error: true, content: "Failed to reach the server." });
          return;
        }

        let text = "";
        for await (const frame of parseSSE(res.body)) {
          if (frame.event === "token") {
            text += (JSON.parse(frame.data) as { text: string }).text;
            patch({ content: text });
          } else if (frame.event === "final") {
            const data = JSON.parse(frame.data) as FinalEventData;
            patch({
              pending: false,
              citations: data.citations,
              insufficientEvidence: data.insufficient_evidence,
            });
          } else if (frame.event === "error") {
            const data = JSON.parse(frame.data) as { message: string };
            patch({ pending: false, error: true, content: text || data.message });
          }
        }
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          patch({ pending: false, error: true, content: "Something went wrong while streaming." });
        }
      } finally {
        patch({ pending: false });
        setIsStreaming(false);
        abortRef.current = null;
      }
    },
    [isStreaming],
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setMessages([]);
    setIsStreaming(false);
  }, []);

  return { messages, isStreaming, send, stop, reset };
}
