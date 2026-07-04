# Observability deploy assets

Reproducible definitions for the operational plane (Cloud Logging, Cloud
Monitoring, Cloud Trace, Error Reporting). The log-based metrics, dashboard, and
request alerts are derived from the single `request_summary` structured log
event the API emits per request; the uptime check probes the public site, Cloud
Trace spans are OpenTelemetry exports, and Error Reporting groups error logs.
Everything stays in version control rather than as click-ops in the console.

The LLM/product plane (per-query token cost, the gate -> retrieval (semantic,
bm25, merge, rerank) -> generation trace, prompt versions) lives in Langfuse and
is configured by env vars on the service, not here.

## What it looks like

![Kuberacle - serving dashboard in Cloud Monitoring: KPI strip for requests,
answer-cache hit rate, cost avoided by cache, and cold starts, above daily
charts for requests by outcome, p95 latency, paid-vs-cache-saved cost, and cache
hits vs misses](../../.github/assets/observability-dashboard.png)

*The `Kuberacle - serving` Cloud Monitoring dashboard: a KPI strip (requests,
answer-cache hit rate, cost avoided by cache, cold starts) over daily charts for
requests by outcome, p95 latency, paid-vs-cache-saved cost, and cache hits vs
misses. Defined in `dashboard.json`.*

![Langfuse trace for a single query, showing the gate -> retrieval (semantic,
bm25, merge, rerank) -> generation span tree with per-stage latency and cost
alongside the grounded answer](../../.github/assets/observability-trace.png)

*A single request in Langfuse: each pipeline stage is its own span with its own
latency, and the billed stages carry cost metadata that `cost_usd` totals by
stage (gate, embed, rerank, generation).*

![Langfuse cost view over the same traces, with total cost, cost by model, and
cost by environment](../../.github/assets/observability-llm-cost.png)

*Langfuse's built-in cost view aggregates the same traces, breaking spend down by
model and environment, a per-query cost lens the Cloud Monitoring dashboard does
not provide.*

## Prerequisites

Set these once for the commands below:

```bash
export PROJECT=<your-project-id>
export REGION=us-central1
export SERVICE=k8s-rag-api
```

## 1. Enable APIs and grant trace IAM

```bash
gcloud services enable monitoring.googleapis.com cloudtrace.googleapis.com \
  logging.googleapis.com secretmanager.googleapis.com \
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

# Create a notification channel for the ops-alert email, then capture its id.
export NOTIFICATION_CHANNEL=$(gcloud beta monitoring channels create \
  --project "$PROJECT" --display-name="kuberacle ops email" \
  --type=email --channel-labels=email_address=<your-ops-alert-email> \
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
- **Alerts:** error-rate, per-request cost anomaly, p95 latency, and uptime.
- **Error Reporting:** API exceptions group automatically from the ERROR-severity
  stack traces the structured logger emits.

## Cost

All of this stays within GCP free tiers at the 300/day global cap (Cloud Logging
50 GiB/mo ingest, Cloud Trace 2.5M spans/mo, and the user-defined log-based
metrics within Monitoring's free metric allotment). Langfuse runs on its free
Hobby tier.
