/** Proxy the browser's query to the FastAPI backend, streaming SSE through. */

export const dynamic = "force-dynamic";

const API_URL = process.env.RAG_API_URL ?? "http://127.0.0.1:8000";

function sseError(message: string, status: number): Response {
  return new Response(`event: error\ndata: ${JSON.stringify({ message })}\n\n`, {
    status,
    headers: { "Content-Type": "text/event-stream" },
  });
}

export async function POST(request: Request): Promise<Response> {
  const body = await request.text();

  let upstream: Response;
  try {
    upstream = await fetch(`${API_URL}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      signal: request.signal,
    });
  } catch {
    return sseError("Could not reach the RAG API. Is the backend running?", 502);
  }

  if (!upstream.ok || !upstream.body) {
    return sseError("The RAG API returned an error.", 502);
  }

  return new Response(upstream.body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
