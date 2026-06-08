"use client";

import { useEffect, useRef } from "react";

import { ChatInput } from "@/components/chat-input";
import { EmptyState } from "@/components/empty-state";
import { Message } from "@/components/message";
import { TopBar } from "@/components/top-bar";
import { useRagChat } from "@/lib/use-rag-chat";
import type { ChatMessage } from "@/lib/types";

/** Group the flat message list into question/answer turns for the canvas. */
function toTurns(messages: ChatMessage[]) {
  const turns: { user: ChatMessage; assistant?: ChatMessage }[] = [];
  for (const message of messages) {
    if (message.role === "user") {
      turns.push({ user: message });
    } else if (turns.length > 0) {
      turns[turns.length - 1].assistant = message;
    }
  }
  return turns;
}

export function Chat() {
  const { messages, isStreaming, send, stop } = useRagChat();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const isEmpty = messages.length === 0;

  return (
    <div className="flex h-full w-full flex-col">
      <TopBar />

      {isEmpty ? (
        <div className="flex-1 overflow-y-auto">
          <EmptyState onSend={send} />
        </div>
      ) : (
        <div className="mx-auto flex min-h-0 w-full max-w-3xl flex-1 flex-col">
          <div className="flex-1 overflow-y-auto px-4 py-8">
            <div className="flex flex-col gap-14">
              {toTurns(messages).map((turn) => (
                <div key={turn.user.id} className="flex flex-col gap-5">
                  <Message message={turn.user} />
                  {turn.assistant && <Message message={turn.assistant} />}
                </div>
              ))}
              <div ref={bottomRef} />
            </div>
          </div>

          <div className="bg-[linear-gradient(to_top,var(--background)_60%,transparent)] px-4 pb-5 pt-3">
            <ChatInput
              onSend={send}
              onStop={stop}
              isStreaming={isStreaming}
              placeholder="Ask a follow-up…"
            />
            <p className="mt-2 text-center text-[11px] text-text-3">
              Answers may be incomplete. Always verify against the official docs.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
