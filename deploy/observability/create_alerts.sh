#!/usr/bin/env bash
# Create alert policies on the log-based metrics.
# Requires: PROJECT and NOTIFICATION_CHANNEL env vars (see README).
#
# Policies are defined via --policy-from-file (JSON), which is stable across
# gcloud versions; the --condition-threshold-* flags are not on the GA command.
set -euo pipefail

: "${PROJECT:?set PROJECT}"
: "${NOTIFICATION_CHANNEL:?set NOTIFICATION_CHANNEL (channel resource name)}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# 1. Error-rate alert: any error-outcome requests over 5 minutes.
cat > "$TMP/errors.json" <<JSON
{
  "displayName": "Kuberacle: request errors",
  "combiner": "OR",
  "conditions": [{
    "displayName": "error outcomes > 0 (5m)",
    "conditionThreshold": {
      "filter": "metric.type=\"logging.googleapis.com/user/kuberacle_requests\" resource.type=\"cloud_run_revision\" metric.label.outcome=\"error\"",
      "comparison": "COMPARISON_GT",
      "thresholdValue": 0,
      "duration": "300s",
      "aggregations": [{"alignmentPeriod": "300s", "perSeriesAligner": "ALIGN_SUM"}]
    }
  }],
  "notificationChannels": ["${NOTIFICATION_CHANNEL}"]
}
JSON

# 2. p95 latency alert: 95th percentile request latency above 8s over 10 minutes.
cat > "$TMP/latency.json" <<JSON
{
  "displayName": "Kuberacle: p95 latency",
  "combiner": "OR",
  "conditions": [{
    "displayName": "p95 duration > 8000ms (10m)",
    "conditionThreshold": {
      "filter": "metric.type=\"logging.googleapis.com/user/kuberacle_request_latency_ms\" resource.type=\"cloud_run_revision\"",
      "comparison": "COMPARISON_GT",
      "thresholdValue": 8000,
      "duration": "600s",
      "aggregations": [{"alignmentPeriod": "600s", "perSeriesAligner": "ALIGN_PERCENTILE_95"}]
    }
  }],
  "notificationChannels": ["${NOTIFICATION_CHANNEL}"]
}
JSON

# 3. Per-request cost anomaly: a single query far above normal (~$0.0017) points
#    to a runaway-token bug or abuse. Total daily spend is already bounded by the
#    budget alert + the 300/day global cap, so this catches the per-request case
#    a distribution metric can actually express (p99 needs a percentile aligner).
cat > "$TMP/cost.json" <<JSON
{
  "displayName": "Kuberacle: per-request cost anomaly",
  "combiner": "OR",
  "conditions": [{
    "displayName": "p99 per-request cost > 0.02 USD",
    "conditionThreshold": {
      "filter": "metric.type=\"logging.googleapis.com/user/kuberacle_request_cost_usd\" resource.type=\"cloud_run_revision\"",
      "comparison": "COMPARISON_GT",
      "thresholdValue": 0.02,
      "duration": "0s",
      "aggregations": [{"alignmentPeriod": "600s", "perSeriesAligner": "ALIGN_PERCENTILE_99"}]
    }
  }],
  "notificationChannels": ["${NOTIFICATION_CHANNEL}"]
}
JSON

for f in errors latency cost; do
  gcloud alpha monitoring policies create --project "$PROJECT" \
    --policy-from-file="$TMP/$f.json"
done

echo "Created alert policies: errors, p95 latency, per-request cost anomaly"
