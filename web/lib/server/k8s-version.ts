/** Server-side fetch of the served index's Kubernetes docs version.
 *
 * Reads it from the API's /meta endpoint at request time so a docs-version bump
 * (a new index) is reflected without rebuilding the web image. Authenticates to
 * the private API with an OIDC identity token, mirroring the query proxy.
 */

import { GoogleAuth } from "google-auth-library";

import { DEFAULT_K8S_VERSION } from "@/lib/constants";

const API_URL = process.env.RAG_API_URL ?? "http://127.0.0.1:8000";
const API_AUDIENCE = process.env.RAG_API_AUDIENCE;

const auth = new GoogleAuth();

async function authHeader(): Promise<Record<string, string>> {
  if (!API_AUDIENCE) return {};
  const client = await auth.getIdTokenClient(API_AUDIENCE);
  const token = await client.idTokenProvider.fetchIdToken(API_AUDIENCE);
  return { Authorization: `Bearer ${token}` };
}

export async function getK8sVersion(): Promise<string> {
  try {
    const headers = await authHeader();
    // Cache briefly so navigation does not hit the API on every render; a new
    // index changes the version at most weekly.
    const res = await fetch(`${API_URL}/meta`, {
      headers,
      next: { revalidate: 3600 },
    });
    if (!res.ok) return DEFAULT_K8S_VERSION;
    const data = (await res.json()) as { k8s_version?: unknown };
    return typeof data.k8s_version === "string" && data.k8s_version
      ? data.k8s_version
      : DEFAULT_K8S_VERSION;
  } catch {
    return DEFAULT_K8S_VERSION;
  }
}
