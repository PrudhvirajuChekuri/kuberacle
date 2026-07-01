# Observability deploy assets

Reproducible definitions for the operational plane (Cloud Logging, Cloud
Monitoring, Cloud Trace, Error Reporting). Everything here is derived from the
single `request_summary` structured log event the API emits per request, so the
metrics, dashboard, and alerts stay in version control rather than as
click-ops in the console.

The LLM/product plane (per-query token cost, the retrieval -> rerank ->
generation trace, prompt versions) lives in Langfuse and is configured by env
vars on the service, not here.

## Prerequisites

Set these once for the commands below:

```bash
export PROJECT=project-d8548532-f717-4b4d-a95
export REGION=us-central1
export SERVICE=k8s-rag-api
```

## 1. Enable APIs and grant trace IAM

```bash
gcloud services enable monitoring.googleapis.com cloudtrace.googleapis.com \
  --project "$PROJECT"

# The API service account exports spans to Cloud Trace.
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:api-sa@${PROJECT}.iam.gserviceaccount.com" \
  --role="roles/cloudtrace.agent"
```

## 2. Set the API service env (enables tracing + Langfuse)

`OBSERVABILITY_ENABLED` is the master switch. Langfuse keys are secrets; store
them in Secret Manager and reference them, never inline (see the no-keys rule).

```bash
# One-time: store the Langfuse keys as secrets (paste when prompted).
printf '%s' '<langfuse-public-key>' | gcloud secrets create langfuse-public-key \
  --data-file=- --project "$PROJECT"
printf '%s' '<langfuse-secret-key>' | gcloud secrets create langfuse-secret-key \
  --data-file=- --project "$PROJECT"
gcloud secrets add-iam-policy-binding langfuse-public-key --project "$PROJECT" \
  --member="serviceAccount:api-sa@${PROJECT}.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
gcloud secrets add-iam-policy-binding langfuse-secret-key --project "$PROJECT" \
  --member="serviceAccount:api-sa@${PROJECT}.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud run services update "$SERVICE" --region "$REGION" --project "$PROJECT" \
  --update-env-vars="OBSERVABILITY_ENABLED=true,OBSERVABILITY_ENVIRONMENT=production,SERVICE_VERSION=$(git rev-parse --short HEAD),LANGFUSE_HOST=https://us.cloud.langfuse.com" \
  --update-secrets="LANGFUSE_PUBLIC_KEY=langfuse-public-key:latest,LANGFUSE_SECRET_KEY=langfuse-secret-key:latest"
```

After deploy, push the file prompts to Langfuse so they serve as managed
(the files remain the fallback and the source of truth):

```bash
OBSERVABILITY_ENABLED=true LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... \
  python -m kuberacle sync-prompts
```

## 3. Create the log-based metrics, dashboard, alerts, and uptime check

```bash
bash deploy/observability/create_log_metrics.sh
gcloud monitoring dashboards create \
  --project "$PROJECT" --config-from-file=deploy/observability/dashboard.json

# Create a notification channel for the budget-alert email, then capture its id.
export NOTIFICATION_CHANNEL=$(gcloud beta monitoring channels create \
  --project "$PROJECT" --display-name="kuberacle ops email" \
  --type=email --channel-labels=email_address=<your-budget-alert-email> \
  --format='value(name)')

bash deploy/observability/create_alerts.sh
bash deploy/observability/create_uptime_check.sh
```

## What you get

- **Logs:** structured JSON in Cloud Logging, each request line trace-correlated
  to its Cloud Trace span. Query: `jsonPayload.event="request_summary"`.
- **Metrics (log-based):** request count by outcome, p50/p95 latency, estimated
  cost per request, abstention/error rates, answer-cache requests by
  `cache_hit`, and estimated cost avoided by cache hits.
- **Dashboard:** `Kuberacle - serving` - a KPI strip (requests, answer-cache
  hit-rate gauge, cost avoided by cache, cold starts) over daily charts
  (requests by outcome, p95 latency, daily paid-vs-cache-saved cost, cache hits
  vs misses). Requests, hit rate, and cold starts follow the dashboard
  time-picker via filter-based queries with `outputFullDuration`. The
  cost-avoided scorecard stays on MQL, because its saved-cost metric is a
  DISTRIBUTION and only MQL `sum()` reduces a distribution to a scalar dollar
  total; MQL scorecards do not honor `outputFullDuration`, so it is pinned to a
  rolling 30-day window (reflected in its `(30d)` title). The daily charts
  align at day granularity and follow the picker; the paid-vs-cache-saved chart
  stacks two bar datasets so bar height reads as the estimated no-cache cost.
- **Alerts:** error-rate, estimated daily cost ceiling, p95 latency, and uptime.
- **Error Reporting:** API exceptions group automatically from the ERROR-severity
  stack traces the structured logger emits.

## Cost

All of this stays within GCP free tiers at the 300/day global cap (Cloud Logging
50 GiB/mo ingest, Cloud Trace 2.5M spans/mo, log-based metrics free). Langfuse
runs on its free Hobby tier.
