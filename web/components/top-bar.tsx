"use client";

import { Brand } from "@/components/brand";
import { ThemeToggle } from "@/components/theme-toggle";
import { useK8sVersion } from "@/components/version-provider";

/** Full-width application header: brand, docs version, theme toggle. */
export function TopBar({ onHome }: { onHome?: () => void }) {
  const k8sVersion = useK8sVersion();
  return (
    <header className="relative z-20 flex h-[60px] flex-none items-center justify-between border-b border-border bg-background/80 px-5 backdrop-blur-md">
      <Brand onHome={onHome} />
      <div className="flex items-center gap-2.5">
        <div className="flex items-center gap-1.5 rounded-lg border border-border bg-card px-2.5 py-1.5 font-mono text-xs text-muted-foreground">
          <span className="h-1.5 w-1.5 rounded-full bg-[#21c07a] shadow-[0_0_0_3px_color-mix(in_srgb,#21c07a_22%,transparent)]" />
          {k8sVersion}
        </div>
        <ThemeToggle />
      </div>
    </header>
  );
}
