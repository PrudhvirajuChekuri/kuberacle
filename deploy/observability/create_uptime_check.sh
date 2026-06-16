#!/usr/bin/env bash
# Create an uptime check on the public site and alert on failure.
# Requires: PROJECT and NOTIFICATION_CHANNEL env vars (see README).
set -euo pipefail

: "${PROJECT:?set PROJECT}"
: "${NOTIFICATION_CHANNEL:?set NOTIFICATION_CHANNEL (channel resource name)}"

# Uptime check against the public web service home page.
gcloud monitoring uptime create kuberacle-home \
  --project "$PROJECT" \
  --resource-type=uptime-url \
  --resource-labels=host=kuberacle.dev \
  --path="/" \
  --port=443 \
  --protocol=https \
  --period=5

# Alert when the uptime check fails.
gcloud alpha monitoring policies create --project "$PROJECT" \
  --notification-channels="$NOTIFICATION_CHANNEL" \
  --display-name="Kuberacle: site down" \
  --condition-display-name="uptime check failing" \
  --condition-filter='metric.type="monitoring.googleapis.com/uptime_check/check_passed" resource.type="uptime_url" metric.label.check_id=starts_with("kuberacle-home")' \
  --condition-threshold-value=1 \
  --condition-threshold-comparison=COMPARISON_LT \
  --condition-threshold-duration=300s \
  --aggregation='{"alignmentPeriod":"300s","perSeriesAligner":"ALIGN_FRACTION_TRUE"}'

echo "Created uptime check kuberacle-home and its alert"
