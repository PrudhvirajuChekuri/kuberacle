#!/usr/bin/env bash
# Create an uptime check on the public site and alert on failure.
# Requires: PROJECT and NOTIFICATION_CHANNEL env vars (see README).
#
# The alert policy is defined via --policy-from-file (JSON), stable across
# gcloud versions; the --condition-threshold-* flags are not on the GA command.
set -euo pipefail

: "${PROJECT:?set PROJECT}"
: "${NOTIFICATION_CHANNEL:?set NOTIFICATION_CHANNEL (channel resource name)}"

# Uptime check against the public web service home page.
gcloud monitoring uptime create kuberacle-home \
  --project "$PROJECT" \
  --resource-type=uptime-url \
  --resource-labels=host=kuberacle.dev,project_id="$PROJECT" \
  --path="/" \
  --port=443 \
  --protocol=https \
  --period=5

# Alert when the uptime check fails.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/uptime.json" <<JSON
{
  "displayName": "Kuberacle: site down",
  "combiner": "OR",
  "conditions": [{
    "displayName": "uptime check failing",
    "conditionThreshold": {
      "filter": "metric.type=\"monitoring.googleapis.com/uptime_check/check_passed\" resource.type=\"uptime_url\"",
      "comparison": "COMPARISON_LT",
      "thresholdValue": 1,
      "duration": "300s",
      "aggregations": [{"alignmentPeriod": "300s", "perSeriesAligner": "ALIGN_FRACTION_TRUE"}]
    }
  }],
  "notificationChannels": ["${NOTIFICATION_CHANNEL}"]
}
JSON
gcloud alpha monitoring policies create --project "$PROJECT" \
  --policy-from-file="$TMP/uptime.json"

echo "Created uptime check kuberacle-home and its alert"
