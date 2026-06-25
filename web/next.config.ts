import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  // Drop the framework-fingerprinting X-Powered-By response header.
  poweredByHeader: false,
  // The corpus Kubernetes version is fetched at runtime from the API's /meta
  // endpoint (see lib/server/k8s-version.ts), not baked at build time.
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
