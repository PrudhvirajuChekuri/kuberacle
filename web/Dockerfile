# Frontend image for the Kuberacle Next.js UI.
# Build context is the repo root (standalone layout expects web/ under /app).
# Multi-stage: build the standalone bundle, then serve it on a slim node runtime.

FROM node:22-alpine AS builder

WORKDIR /app

# Install dependencies against the lockfile for reproducible builds.
COPY web/package.json web/package-lock.json ./
RUN npm ci

# App source. The corpus version is fetched from the API at runtime, so no
# data/ files are needed at build time.
COPY web/ ./

# Public Turnstile site key, baked into the client bundle at build time.
# Unset (e.g. local compose) leaves Turnstile disabled in the built image.
ARG NEXT_PUBLIC_TURNSTILE_SITE_KEY
ENV NEXT_PUBLIC_TURNSTILE_SITE_KEY=$NEXT_PUBLIC_TURNSTILE_SITE_KEY

# Produces .next/standalone (requires output: "standalone" in next.config.ts).
RUN npm run build

FROM node:22-alpine AS runtime

WORKDIR /app
ENV NODE_ENV=production

# Standalone server bundle plus the static assets it does not inline.
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000

# Run as the image's built-in non-root user; the bundle is read-only at runtime.
USER node

# Next standalone emits server.js at the app root.
CMD ["node", "server.js"]
