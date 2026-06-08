/** Minimal Server-Sent Events parser over a fetch response body stream. */

export interface SSEFrame {
  event: string;
  data: string;
}

/**
 * Parse a byte stream of SSE frames into `{ event, data }` objects.
 *
 * Frames are separated by a blank line; `event:` and `data:` fields are
 * recognized. Frames without a data field are skipped.
 */
export async function* parseSSE(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<SSEFrame> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary: number;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const rawFrame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const frame = parseFrame(rawFrame);
        if (frame) yield frame;
      }
    }
    buffer += decoder.decode();
    const frame = parseFrame(buffer);
    if (frame) yield frame;
  } finally {
    reader.releaseLock();
  }
}

function parseFrame(raw: string): SSEFrame | null {
  let event = "message";
  const dataLines: string[] = [];

  for (const rawLine of raw.split("\n")) {
    const line = rawLine.replace(/\r$/, "");
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }

  if (dataLines.length === 0) return null;
  return { event, data: dataLines.join("\n") };
}
