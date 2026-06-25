/**
 * Fallback Kubernetes docs version, shown only if the API's /meta is
 * unreachable. The live value is the served index's version, fetched at request
 * time (see lib/server/k8s-version.ts) and provided via K8sVersionProvider.
 */
export const DEFAULT_K8S_VERSION = "v1.36";

/**
 * Abstention sentinel emitted by the backend (mirrors ABSTENTION_SENTINEL in
 * kuberacle/constants.py). The `final` SSE event carries an authoritative
 * `abstained` flag; this is only used to bridge the streaming window before that
 * flag arrives, so a raw sentinel answer is not briefly shown as a real answer.
 */
export const ABSTENTION_SENTINEL = "INSUFFICIENT_EVIDENCE";
