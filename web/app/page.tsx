import { Chat } from "@/components/chat";
import { TurnstileProvider } from "@/components/turnstile";

export default function Home() {
  return (
    <div className="h-dvh bg-background">
      <TurnstileProvider>
        <Chat />
      </TurnstileProvider>
    </div>
  );
}
