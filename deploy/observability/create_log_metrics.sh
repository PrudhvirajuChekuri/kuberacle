#!/usr/bin/env bash
# Create log-based metrics derived from the request_summary event.
# Requires: PROJECT env var. Idempotent-ish: re-running errors on existing
# metrics; delete first with `gcloud logging metrics delete <name>` to recreate.
set -euo pipefail

: "${PROJECT:?set PROJECT}"

BASE_FILTER='resource.type="cloud_run_revision" jsonPayload.event="request_summary"'

# Request count, labelled by outcome and cold_start (for RED + abstention rates).
gcloud logging metrics create kuberacle_requests \
  --project "$PROJECT" \
  --description="Kuberacle requests by outcome" \
  --log-filter="$BASE_FILTER" \
  --label-extractors='outcome=EXTRACT(jsonPayload.outcome),cold_start=EXTRACT(jsonPayload.cold_start)' \
  --metric-descriptor='{"metricKind":"DELTA","valueType":"INT64","labels":[{"key":"outcome"},{"key":"cold_start"}]}'

# Request latency distribution (ms) for p50/p95.
gcloud logging metrics create kuberacle_request_latency_ms \
  --project "$PROJECT" \
  --description="Kuberacle request duration (ms)" \
  --log-filter="$BASE_FILTER" \
  --value-extractor='EXTRACT(jsonPayload.duration_ms)' \
  --metric-descriptor='{"metricKind":"DELTA","valueType":"DISTRIBUTION","unit":"ms"}'

# Estimated cost per request (USD), summed for the daily-cost alert.
gcloud logging metrics create kuberacle_request_cost_usd \
  --project "$PROJECT" \
  --description="Kuberacle estimated cost per request (USD)" \
  --log-filter="$BASE_FILTER" \
  --value-extractor='EXTRACT(jsonPayload.cost_usd.total)' \
  --metric-descriptor='{"metricKind":"DELTA","valueType":"DISTRIBUTION","unit":"USD"}'

echo "Created log-based metrics: kuberacle_requests, kuberacle_request_latency_ms, kuberacle_request_cost_usd"
