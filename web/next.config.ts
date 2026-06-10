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
  // Baseline hardening headers. Transport is covered by the .dev HSTS preload.
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Frame-Options", value: "DENY" },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
        ],
      },
    ];
  },
};

export default nextConfig;
