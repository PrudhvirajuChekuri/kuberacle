/** Corpus Kubernetes version, sourced from data/k8s_version.txt at build time. */
export const K8S_VERSION = process.env.NEXT_PUBLIC_K8S_VERSION ?? "v1.36";
