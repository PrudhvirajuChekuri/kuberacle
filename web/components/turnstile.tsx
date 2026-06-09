"use client";

/**
 * Cloudflare Turnstile provider.
 *
 * Renders a single explicit-execution widget and exposes `getToken()`, which
 * mints a fresh, single-use token per call (token-per-query). Turnstile is
 * enabled only when `NEXT_PUBLIC_TURNSTILE_SITE_KEY` is set at build time; with
 * it unset (local dev) the provider is inert and callers skip verification.
 */

import Script from "next/script";
import {
  createContext,
  useCallback,
  useContext,
  useRef,
  type ReactNode,
} from "react";

interface RenderOptions {
  sitekey: string;
  execution?: "render" | "execute";
  appearance?: "always" | "execute" | "interaction-only";
  callback?: (token: string) => void;
  "error-callback"?: () => void;
  "expired-callback"?: () => void;
}

interface TurnstileAPI {
  render: (container: string | HTMLElement, options: RenderOptions) => string;
  execute: (container: string | HTMLElement) => void;
  reset: (widgetId?: string) => void;
  getResponse: (widgetId?: string) => string | undefined;
  remove: (widgetId?: string) => void;
}

declare global {
  interface Window {
    turnstile?: TurnstileAPI;
  }
}

interface TurnstileContextValue {
  /** Whether Turnstile is configured and should be used. */
  enabled: boolean;
  /** Resolve a fresh single-use token, or reject if verification fails. */
  getToken: () => Promise<string>;
}

const TurnstileContext = createContext<TurnstileContextValue | null>(null);

const SITE_KEY = process.env.NEXT_PUBLIC_TURNSTILE_SITE_KEY;

type Resolver = { resolve: (token: string) => void; reject: (error: Error) => void };

export function TurnstileProvider({ children }: { children: ReactNode }) {
  const enabled = Boolean(SITE_KEY);

  const containerRef = useRef<HTMLDivElement>(null);
  const widgetIdRef = useRef<string | null>(null);
  const resolverRef = useRef<Resolver | null>(null);
  const executedRef = useRef(false);

  const settle = (fn: (resolver: Resolver) => void) => {
    const resolver = resolverRef.current;
    resolverRef.current = null;
    if (resolver) fn(resolver);
  };

  const renderWidget = useCallback(() => {
    if (!SITE_KEY || !window.turnstile || !containerRef.current) return;
    if (widgetIdRef.current) return;
    widgetIdRef.current = window.turnstile.render(containerRef.current, {
      sitekey: SITE_KEY,
      execution: "execute",
      appearance: "interaction-only",
      callback: (token) => settle((r) => r.resolve(token)),
      "error-callback": () =>
        settle((r) => r.reject(new Error("Verification failed. Please try again."))),
      "expired-callback": () =>
        settle((r) => r.reject(new Error("Verification expired. Please try again."))),
    });
  }, []);

  const getToken = useCallback(() => {
    return new Promise<string>((resolve, reject) => {
      const api = window.turnstile;
      const widgetId = widgetIdRef.current;
      if (!api || !widgetId) {
        reject(new Error("Verification is still loading. Please try again."));
        return;
      }
      resolverRef.current = { resolve, reject };
      // Reset before re-running so each query gets a fresh, unused token.
      if (executedRef.current) api.reset(widgetId);
      executedRef.current = true;
      api.execute(widgetId);
    });
  }, []);

  return (
    <TurnstileContext.Provider value={{ enabled, getToken }}>
      {children}
      <div
        ref={containerRef}
        className="fixed bottom-4 left-1/2 z-50 -translate-x-1/2"
      />
      {enabled && (
        <Script
          src="https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit"
          strategy="afterInteractive"
          onLoad={renderWidget}
        />
      )}
    </TurnstileContext.Provider>
  );
}

/** Access the Turnstile context (null when no provider is mounted). */
export function useTurnstile(): TurnstileContextValue | null {
  return useContext(TurnstileContext);
}
