/** Shared types for the RAG chat client. */

/** A source citation returned with a grounded answer. */
export interface Citation {
  /** 1-based marker number used in the answer (`[n]`). */
  index: number;
  source_url: string;
  chunk_id: string;
  score: number;
  /** Document title of the supporting chunk, for source previews. */
  title: string;
  /** Short text preview of the supporting chunk content. */
  snippet: string;
}

/** A single chat message in the conversation. */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  /** Validated citations attached to a completed assistant answer. */
  citations?: Citation[];
  /** True when the answer could not be grounded in any verified source. */
  insufficientEvidence?: boolean;
  /** True when the request failed before/while streaming. */
  error?: boolean;
  /** True while the assistant answer is still streaming. */
  pending?: boolean;
}

/** Shape of the `final` SSE event payload from the backend. */
export interface FinalEventData {
  citations: Citation[];
  insufficient_evidence: boolean;
}
