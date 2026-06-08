import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { NextConfig } from "next";

/** Read the corpus Kubernetes version from the repo's single source of truth. */
function readK8sVersion(): string {
  try {
    return readFileSync(join(process.cwd(), "..", "data", "k8s_version.txt"), "utf8").trim();
  } catch {
    return "v1.36";
  }
}

const nextConfig: NextConfig = {
  output: "standalone",
  env: {
    NEXT_PUBLIC_K8S_VERSION: readK8sVersion(),
  },
};

export default nextConfig;
