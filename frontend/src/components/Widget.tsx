import { useChat } from "../hooks/useChat";
import { Header } from "./Header";
import { MessageList } from "./MessageList";
import { Composer } from "./Composer";

export function Widget() {
  const chat = useChat();

  return (
    <div className="lamp-window flex flex-col overflow-hidden rounded-2xl border border-[0.5px] border-border bg-bg">
      <Header />
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
