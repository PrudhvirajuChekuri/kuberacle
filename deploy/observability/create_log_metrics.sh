#!/usr/bin/env bash
# Create log-based metrics derived from the request_summary event.
# Requires: PROJECT env var. Re-running errors on existing metrics; delete first
# with `gcloud logging metrics delete <name>` to recreate.
#
# Complex log metrics (labels, distributions, value extractors) are defined via
# --config-from-file, which is stable across gcloud versions; the equivalent
# flags are not available on the GA `gcloud logging metrics create`.
set -euo pipefail

: "${PROJECT:?set PROJECT}"

BASE_FILTER='resource.type="cloud_run_revision" jsonPayload.event="request_summary"'
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Request count, labelled by outcome and cold_start (RED + abstention rates).
cat > "$TMP/requests.yaml" <<YAML
description: "Kuberacle requests by outcome"
filter: '${BASE_FILTER}'
metricDescriptor:
  metricKind: DELTA
  valueType: INT64
  labels:
  - key: outcome
  - key: cold_start
labelExtractors:
  outcome: EXTRACT(jsonPayload.outcome)
  cold_start: EXTRACT(jsonPayload.cold_start)
YAML

# Request latency distribution (ms) for p50/p95.
cat > "$TMP/latency.yaml" <<YAML
description: "Kuberacle request duration (ms)"
filter: '${BASE_FILTER}'
metricDescriptor:
  metricKind: DELTA
  valueType: DISTRIBUTION
  unit: ms
valueExtractor: EXTRACT(jsonPayload.duration_ms)
bucketOptions:
  exponentialBuckets:
    numFiniteBuckets: 64
    growthFactor: 1.4
    scale: 1
YAML

# Estimated cost per request (USD), summed for the daily-cost alert.
cat > "$TMP/cost.yaml" <<YAML
description: "Kuberacle estimated cost per request (USD)"
filter: '${BASE_FILTER}'
metricDescriptor:
  metricKind: DELTA
  valueType: DISTRIBUTION
  unit: "1"
valueExtractor: EXTRACT(jsonPayload.cost_usd.total)
bucketOptions:
  exponentialBuckets:
    numFiniteBuckets: 64
    growthFactor: 1.4
    scale: 0.0001
YAML

# Answer-cache requests, labelled by cache_hit (numerator + denominator for the
# hit-rate ratio). Boolean cache_hit extracts to the strings "true"/"false".
cat > "$TMP/cache_requests.yaml" <<YAML
description: "Kuberacle answer-cache requests by cache_hit"
filter: '${BASE_FILTER}'
metricDescriptor:
  metricKind: DELTA
  valueType: INT64
  labels:
  - key: cache_hit
labelExtractors:
  cache_hit: EXTRACT(jsonPayload.cache_hit)
YAML

# Estimated pipeline cost avoided by cache hits (USD), summed per day. Misses
# log 0.0 and contribute nothing to the sum.
cat > "$TMP/saved_cost.yaml" <<YAML
description: "Kuberacle estimated cost avoided by answer cache (USD)"
filter: '${BASE_FILTER}'
metricDescriptor:
  metricKind: DELTA
  valueType: DISTRIBUTION
  unit: "1"
valueExtractor: EXTRACT(jsonPayload.saved_cost_estimate)
bucketOptions:
  exponentialBuckets:
    numFiniteBuckets: 64
    growthFactor: 1.4
    scale: 0.0001
YAML

gcloud logging metrics create kuberacle_requests \
  --project "$PROJECT" --config-from-file="$TMP/requests.yaml"
gcloud logging metrics create kuberacle_request_latency_ms \
  --project "$PROJECT" --config-from-file="$TMP/latency.yaml"
gcloud logging metrics create kuberacle_request_cost_usd \
  --project "$PROJECT" --config-from-file="$TMP/cost.yaml"
gcloud logging metrics create kuberacle_cache_requests \
  --project "$PROJECT" --config-from-file="$TMP/cache_requests.yaml"
gcloud logging metrics create kuberacle_saved_cost_usd \
  --project "$PROJECT" --config-from-file="$TMP/saved_cost.yaml"

echo "Created log-based metrics: kuberacle_requests, kuberacle_request_latency_ms, kuberacle_request_cost_usd, kuberacle_cache_requests, kuberacle_saved_cost_usd"
