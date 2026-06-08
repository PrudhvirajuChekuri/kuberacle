"use client";

import { useSyncExternalStore } from "react";
import { Moon, Sun } from "lucide-react";

/** Subscribe to changes of the `dark` class on <html>. */
function subscribe(callback: () => void) {
  const observer = new MutationObserver(callback);
  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["class"],
  });
  return () => observer.disconnect();
}

const isDark = () => document.documentElement.classList.contains("dark");

/** Toggles the `dark` class on <html> and persists the choice. */
export function ThemeToggle() {
  const dark = useSyncExternalStore(subscribe, isDark, () => true);

  const toggle = () => {
    const next = !dark;
    document.documentElement.classList.toggle("dark", next);
    try {
      localStorage.setItem("theme", next ? "dark" : "light");
    } catch {}
  };

  return (
    <button
      onClick={toggle}
      aria-label="Toggle theme"
      className="grid h-9 w-9 place-items-center rounded-lg border border-border bg-card text-muted-foreground transition-colors hover:border-border-2 hover:text-foreground"
    >
      {dark ? <Sun className="h-[17px] w-[17px]" /> : <Moon className="h-[17px] w-[17px]" />}
    </button>
  );
}
