/** Proxy the browser's query to the FastAPI backend, streaming SSE through. */

import { GoogleAuth } from "google-auth-library";

export const dynamic = "force-dynamic";

const API_URL = process.env.RAG_API_URL ?? "http://127.0.0.1:8000";
// Set in production to the private API's URL; when set, calls are authenticated
// with a service-to-service OIDC identity token. Unset locally (public API).
const API_AUDIENCE = process.env.RAG_API_AUDIENCE;

const auth = new GoogleAuth();

function sseError(message: string, status: number): Response {
  return new Response(`event: error\ndata: ${JSON.stringify({ message })}\n\n`, {
    status,
    headers: { "Content-Type": "text/event-stream" },
  });
}

/** Best-effort client IP from the inbound forwarding headers. */
function clientIp(request: Request): string {
  const forwarded = request.headers.get("x-forwarded-for");
  if (forwarded) {
    const first = forwarded.split(",")[0].trim();
    if (first) return first;
  }
  return request.headers.get("x-real-ip") ?? "";
}

/** Authorization header carrying an OIDC identity token, empty when local. */
async function authHeader(): Promise<Record<string, string>> {
  if (!API_AUDIENCE) return {};
  const client = await auth.getIdTokenClient(API_AUDIENCE);
  const token = await client.idTokenProvider.fetchIdToken(API_AUDIENCE);
  return { Authorization: `Bearer ${token}` };
}

export async function POST(request: Request): Promise<Response> {
  const body = await request.text();

  let authHeaders: Record<string, string>;
  try {
    authHeaders = await authHeader();
  } catch {
    return sseError("Could not authenticate to the RAG API.", 502);
  }

  let upstream: Response;
  try {
    upstream = await fetch(`${API_URL}/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Client-IP": clientIp(request),
        "X-Turnstile-Token": request.headers.get("x-turnstile-token") ?? "",
        ...authHeaders,
      },
      body,
      signal: request.signal,
    });
  } catch {
    return sseError("Could not reach the RAG API. Is the backend running?", 502);
  }

  if (!upstream.body) {
    return sseError("The RAG API returned an error.", 502);
  }

  // Forward the upstream status and body unchanged so guardrail rejections
  // (403/429) reach the browser with their real SSE error message.
  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
