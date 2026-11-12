import type { ReactNode } from "react";
import type { Message } from "../types";
import { ExtractCard } from "./ExtractCard";

/** 轻量 Markdown 渲染器:仅支持 **粗体**、`行内代码`、[链接](url)、换行、无序列表。 */
function renderInline(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) {
      out.push(<strong key={k++}>{tok.slice(2, -2)}</strong>);
    } else if (tok.startsWith("`")) {
      out.push(
        <code key={k++} className="rounded bg-bg px-1">
          {tok.slice(1, -1)}
        </code>,
      );
    } else {
      const lm = /\[([^\]]+)\]\(([^)]+)\)/.exec(tok);
      if (lm) {
        out.push(
          <a
            key={k++}
            href={lm[2]}
            target="_blank"
            rel="noreferrer"
            className="text-accent underline"
          >
            {lm[1]}
          </a>,
        );
      } else {
        out.push(tok);
      }
    }
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

function renderMarkdown(text: string): ReactNode[] {
  const lines = text.split("\n");
  const out: ReactNode[] = [];
  let list: ReactNode[] = [];
  let k = 0;
  const flushList = () => {
    if (list.length) {
      out.push(
        <ul key={`ul-${k++}`} className="my-1 list-disc pl-4">
          {list}
        </ul>,
      );
      list = [];
    }
  };
  for (const line of lines) {
    const li = /^\s*[-*]\s+(.+)$/.exec(line);
    if (li) {
      list.push(<li key={k++}>{renderInline(li[1])}</li>);
      continue;
    }
    flushList();
    if (line === "") {
      out.push(<br key={k++} />);
    } else {
      out.push(<span key={k++}>{renderInline(line)}</span>);
      out.push(<br key={k++} />);
    }
  }
  flushList();
  return out;
}

interface MessageItemProps {
  message: Message;
  isStreaming: boolean;
  onRetry?: () => void;
}

export function MessageItem({ message, isStreaming, onRetry }: MessageItemProps) {
  if (message.role === "extract") {
    return <ExtractCard card={message.card} />;
  }

  if (message.role === "system") {
    return (
      <div className="flex justify-center">
        <div
          className={`max-w-[85%] rounded-xl px-3 py-1.5 text-xs ${
            message.retryable ? "text-danger" : "text-text-dim"
          }`}
        >
          <span>{message.content}</span>
          {message.retryable && onRetry && (
            <button
              onClick={onRetry}
              className="ml-2 underline underline-offset-2 hover:text-text"
            >
              重试
            </button>
          )}
        </div>
      </div>
    );
  }

  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl bg-accent px-3.5 py-2.5 text-sm leading-relaxed text-accent-ink">
          {renderMarkdown(message.content)}
        </div>
      </div>
    );
  }

  // assistant
  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] rounded-2xl bg-bubble-bot px-3.5 py-2.5 text-sm leading-relaxed text-text">
        {message.tool && (
          <div className="mb-1.5 flex w-fit items-center gap-1 rounded-full bg-bg px-2 py-0.5 text-xs text-accent">
            <span aria-hidden="true">🔧</span>
            <span>{message.tool.label}</span>
          </div>
        )}
        {renderMarkdown(message.content)}
        {isStreaming && (
          <span className="cursor-blink ml-0.5 inline-block h-4 w-0.5 translate-y-0.5 bg-accent" />
        )}
        {message.aborted && (
          <span className="ml-2 text-xs text-text-dim">已停止</span>
        )}
      </div>
    </div>
  );
}
