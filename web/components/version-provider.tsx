"use client";

import { createContext, useContext, type ReactNode } from "react";

import { DEFAULT_K8S_VERSION } from "@/lib/constants";

const K8sVersionContext = createContext<string>(DEFAULT_K8S_VERSION);

/** Provides the served index's docs version to client components. */
export function K8sVersionProvider({
  value,
  children,
}: {
  value: string;
  children: ReactNode;
}) {
  return (
    <K8sVersionContext.Provider value={value}>{children}</K8sVersionContext.Provider>
  );
}

/** Read the Kubernetes docs version of the currently served index. */
export function useK8sVersion(): string {
  return useContext(K8sVersionContext);
}
