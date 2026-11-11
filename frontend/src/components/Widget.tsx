import { useState } from "react";
import { useChat } from "../hooks/useChat";
import { Header } from "./Header";
import { MessageList } from "./MessageList";
import { Composer } from "./Composer";

function ChatIcon() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M21 11.5a8.5 8.5 0 0 1-8.5 8.5c-1.5 0-2.94-.39-4.18-1.08L4 20l1.07-4.32A8.46 8.46 0 0 1 4 11.5 8.5 8.5 0 0 1 12.5 3 8.5 8.5 0 0 1 21 11.5Z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function Widget() {
  const [open, setOpen] = useState(false);
  const chat = useChat();

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        aria-label="打开客服会话"
        className="fixed bottom-4 right-4 z-50 flex h-14 w-14 items-center justify-center rounded-full bg-accent text-accent-ink"
      >
        <ChatIcon />
      </button>
    );
  }

  return (
    <div className="fixed bottom-4 right-4 z-50 flex h-[600px] w-[380px] flex-col overflow-hidden rounded-2xl border border-[0.5px] border-border bg-bg max-sm:inset-0 max-sm:h-full max-sm:w-full max-sm:rounded-none max-sm:border-0">
      <Header onClose={() => setOpen(false)} />
      <MessageList
        messages={chat.messages}
        status={chat.status}
        onRetry={chat.retry}
        onPick={chat.send}
      />
      <Composer
        status={chat.status}
        canExtract={chat.hasUserMessage}
        onSend={chat.send}
        onStop={chat.stop}
        onExtract={chat.extract}
      />
    </div>
  );
}
