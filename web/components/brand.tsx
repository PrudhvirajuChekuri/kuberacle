import { Box } from "lucide-react";

/** Kuberacle wordmark + logo mark for the top bar. */
export function Brand() {
  return (
    <div className="flex items-center gap-3">
      <div className="grid h-9 w-9 place-items-center rounded-[9px] bg-primary text-white shadow-[0_4px_14px_-4px_var(--brand-line)]">
        <Box className="h-5 w-5" strokeWidth={1.6} />
      </div>
      <div className="flex flex-col leading-none">
        <span className="text-base font-bold tracking-tight">
          kube<span className="text-primary">racle</span>
        </span>
        <span className="mt-0.5 font-mono text-[10.5px] text-text-3">
          kubernetes docs assistant
        </span>
      </div>
    </div>
  );
}
