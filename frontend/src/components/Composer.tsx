import { useState } from "react";
import type { ChatStatus } from "../types";

interface ComposerProps {
  status: ChatStatus;
  canExtract: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
  onExtract: () => void;
}

export function Composer({
  status,
  canExtract,
  onSend,
  onStop,
  onExtract,
}: ComposerProps) {
  const [text, setText] = useState("");
  const streaming = status === "sending" || status === "streaming";

  const submit = () => {
    const t = text.trim();
    if (!t || streaming) return;
    onSend(t);
    setText("");
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="border-t border-[0.5px] border-border p-3">
      <div className="mb-2 flex justify-end">
        <button
          onClick={onExtract}
          disabled={!canExtract}
          className="rounded-lg border border-[0.5px] border-border px-2.5 py-1 text-xs text-text-dim transition-colors hover:text-text disabled:opacity-40"
        >
          识别诉求
        </button>
      </div>
      <div className="flex items-end gap-2">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="请输入您的问题…"
          rows={1}
          className="max-h-28 min-h-10 flex-1 resize-none rounded-xl bg-surface px-3 py-2.5 text-sm text-text outline-none placeholder:text-text-dim focus:ring-1 focus:ring-accent"
        />
        {streaming ? (
          <button
            onClick={onStop}
            className="shrink-0 rounded-xl border border-[0.5px] border-border px-4 py-2.5 text-sm text-text transition-colors hover:border-danger hover:text-danger"
          >
            停止生成
          </button>
        ) : (
          <button
            onClick={submit}
            disabled={!text.trim()}
            className="shrink-0 rounded-xl bg-accent px-4 py-2.5 text-sm text-accent-ink disabled:opacity-40"
          >
            发送
          </button>
        )}
      </div>
    </div>
  );
}
