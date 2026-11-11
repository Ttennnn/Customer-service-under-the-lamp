import { useEffect, useRef } from "react";
import type { ChatStatus, Message } from "../types";
import { MessageItem } from "./MessageItem";
import { QuickReplies } from "./QuickReplies";

interface MessageListProps {
  messages: Message[];
  status: ChatStatus;
  onRetry: () => void;
  onPick: (text: string) => void;
}

export function MessageList({
  messages,
  status,
  onRetry,
  onPick,
}: MessageListProps) {
  const ref = useRef<HTMLDivElement>(null);
  const atBottomRef = useRef(true);

  const onScroll = () => {
    const el = ref.current;
    if (!el) return;
    atBottomRef.current =
      el.scrollHeight - el.scrollTop - el.clientHeight < 24;
  };

  useEffect(() => {
    const el = ref.current;
    if (el && atBottomRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages]);

  const isStreaming = status === "streaming" || status === "sending";

  return (
    <div ref={ref} onScroll={onScroll} className="flex-1 overflow-y-auto px-4 py-4">
      {messages.length === 0 ? (
        <div className="flex h-full flex-col justify-end gap-3">
          <div className="w-fit max-w-[85%] rounded-2xl bg-bubble-bot px-3.5 py-2.5 text-sm leading-relaxed text-text">
            你好,我是灯下客服。有什么售后问题可以帮你?
          </div>
          <QuickReplies onPick={onPick} />
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          {messages.map((m, i) => {
            const isLast = i === messages.length - 1;
            const isLastStreaming =
              isStreaming && isLast && m.role === "assistant";
            return (
              <MessageItem
                key={m.id}
                message={m}
                isStreaming={isLastStreaming}
                onRetry={onRetry}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}
