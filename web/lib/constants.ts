/** Corpus Kubernetes version, sourced from data/k8s_version.txt at build time. */
export const K8S_VERSION = process.env.NEXT_PUBLIC_K8S_VERSION ?? "v1.36";

/**
 * Abstention sentinel emitted by the backend (mirrors ABSTENTION_SENTINEL in
 * kuberacle/constants.py). The `final` SSE event carries an authoritative
 * `abstained` flag; this is only used to bridge the streaming window before that
 * flag arrives, so a raw sentinel answer is not briefly shown as a real answer.
 */
export const ABSTENTION_SENTINEL = "INSUFFICIENT_EVIDENCE";
