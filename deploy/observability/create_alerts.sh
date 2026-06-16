#!/usr/bin/env bash
# Create alert policies on the log-based metrics.
# Requires: PROJECT and NOTIFICATION_CHANNEL env vars (see README).
set -euo pipefail

: "${PROJECT:?set PROJECT}"
: "${NOTIFICATION_CHANNEL:?set NOTIFICATION_CHANNEL (channel resource name)}"

# 1. Error-rate alert: any error-outcome requests over 5 minutes.
gcloud alpha monitoring policies create --project "$PROJECT" \
  --notification-channels="$NOTIFICATION_CHANNEL" \
  --display-name="Kuberacle: request errors" \
  --condition-display-name="error outcomes > 0 (5m)" \
  --condition-filter='metric.type="logging.googleapis.com/user/kuberacle_requests" resource.type="cloud_run_revision" metric.label.outcome="error"' \
  --condition-threshold-value=0 \
  --condition-threshold-comparison=COMPARISON_GT \
  --condition-threshold-duration=300s \
  --aggregation='{"alignmentPeriod":"300s","perSeriesAligner":"ALIGN_SUM"}'

# 2. p95 latency alert: 95th percentile request latency above 8s over 10 minutes.
gcloud alpha monitoring policies create --project "$PROJECT" \
  --notification-channels="$NOTIFICATION_CHANNEL" \
  --display-name="Kuberacle: p95 latency" \
  --condition-display-name="p95 duration > 8000ms (10m)" \
  --condition-filter='metric.type="logging.googleapis.com/user/kuberacle_request_latency_ms" resource.type="cloud_run_revision"' \
  --condition-threshold-value=8000 \
  --condition-threshold-comparison=COMPARISON_GT \
  --condition-threshold-duration=600s \
  --aggregation='{"alignmentPeriod":"600s","perSeriesAligner":"ALIGN_PERCENTILE_95"}'

# 3. Daily cost alert: summed estimated cost over 24h above the ceiling (USD).
#    Tune the threshold to your budget; the 300/day cap bounds worst case ~$0.50.
gcloud alpha monitoring policies create --project "$PROJECT" \
  --notification-channels="$NOTIFICATION_CHANNEL" \
  --display-name="Kuberacle: daily cost ceiling" \
  --condition-display-name="estimated cost > \$2 / 24h" \
  --condition-filter='metric.type="logging.googleapis.com/user/kuberacle_request_cost_usd" resource.type="cloud_run_revision"' \
  --condition-threshold-value=2 \
  --condition-threshold-comparison=COMPARISON_GT \
  --condition-threshold-duration=0s \
  --aggregation='{"alignmentPeriod":"86400s","perSeriesAligner":"ALIGN_SUM","crossSeriesReducer":"REDUCE_SUM"}'

echo "Created alert policies: errors, p95 latency, daily cost ceiling"
