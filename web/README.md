# Kuberacle - Kubernetes Docs Web UI

Kuberacle is a Next.js chat interface for the [kuberacle](../README.md) pipeline. It streams
grounded answers token-by-token and renders citations as clickable source cards.

## Stack

- **Next.js** (App Router) + **TypeScript**
- **Tailwind CSS v4** + **shadcn/ui**
- **react-markdown** (+ **remark-gfm**) for answer rendering
- A custom `useRagChat` hook consuming the backend's SSE stream (no AI SDK)

## How it works

- `POST /api/query` is a Route Handler that proxies to the FastAPI backend and
  streams the Server-Sent Events response straight through to the browser.
- The backend URL is read from `RAG_API_URL` (see `.env.local`), defaulting to
  `http://127.0.0.1:8000`.
- Clicking an inline `[n]` citation marker scrolls to and highlights its source
  card, and hovering one shows a source preview; an "ungrounded" notice is shown
  when an answer could not be verified.

## Run locally

The backend must be running first (see the [root README](../README.md#run-the-api)):

```bash
python scripts/serve.py        # backend, from the repo root
```

Then, from this directory:

```bash
npm install                    # first time only
npm run dev                    # http://localhost:3000
```

## Configuration

`web/.env.local`:

```
RAG_API_URL=http://127.0.0.1:8000
```
