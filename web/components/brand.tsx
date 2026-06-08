import { Cube } from "@/components/cube-icon";

/** Kuberacle wordmark + logo mark; clicking it returns to the start screen. */
export function Brand({ onHome }: { onHome?: () => void }) {
  return (
    <button
      type="button"
      onClick={onHome}
      aria-label="Back to start"
      className="flex items-center gap-3 rounded-lg transition-opacity hover:opacity-80"
    >
      <div className="grid h-9 w-9 place-items-center rounded-[9px] bg-primary text-white shadow-[0_4px_14px_-4px_var(--brand-line)]">
        <Cube className="h-5 w-5" strokeWidth={1.6} />
      </div>
      <div className="flex flex-col items-start leading-none">
        <span className="text-base font-bold tracking-tight">
          kube<span className="text-primary">racle</span>
        </span>
        <span className="mt-0.5 font-mono text-[10.5px] text-text-3">
          kubernetes docs assistant
        </span>
      </div>
    </button>
  );
}
