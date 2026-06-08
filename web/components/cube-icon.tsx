/** Double-edged cube mark used for Kuberacle branding (matches the design). */
export function Cube({
  className,
  strokeWidth = 1.6,
}: {
  className?: string;
  strokeWidth?: number;
}) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      <path d="M12 2 4 6.5v9L12 20l8-4.5v-9L12 2Zm0 2.3 5.8 3.3L12 10.9 6.2 7.6 12 4.3ZM6 9.2l5 2.9v5.7l-5-2.8V9.2Zm12 0v5.8l-5 2.8v-5.7l5-2.9Z" />
    </svg>
  );
}
