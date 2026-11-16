import { useState } from "react";
import type { ReactNode } from "react";
import type { CitationSource, FeedbackValue, Message } from "../types";
import { ExtractCard } from "./ExtractCard";

const NO_SOURCES: CitationSource[] = [];

/**
 * 轻量 Markdown 渲染器:仅支持 **粗体**、`行内代码`、[链接](url)、引用编号 [n]、换行、无序列表。
 * 引用编号只有能在 sources 中找到对应 n 时才渲染成按钮,其余按纯文本输出。
 */
function renderInline(
  text: string,
  sources: CitationSource[],
  onCitation?: (n: number) => void,
): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\)|\[\d+\])/g;
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
    } else if (/^\[\d+\]$/.test(tok)) {
      // 引用编号:必须能在本轮 sources 中找到,未知编号保持纯文本
      const n = Number(tok.slice(1, -1));
      if (onCitation && sources.some((s) => s.n === n)) {
        out.push(
          <button
            key={k++}
            type="button"
            onClick={() => onCitation(n)}
            aria-label={`查看来源 ${n}`}
            className="mx-0.5 rounded-full bg-bg px-1.5 py-0.5 align-baseline text-xs font-medium text-accent transition-colors hover:bg-border"
          >
            {tok}
          </button>,
        );
      } else {
        out.push(tok);
      }
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

function renderMarkdown(
  text: string,
  sources: CitationSource[] = NO_SOURCES,
  onCitation?: (n: number) => void,
): ReactNode[] {
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
      list.push(<li key={k++}>{renderInline(li[1], sources, onCitation)}</li>);
      continue;
    }
    flushList();
    if (line === "") {
      out.push(<br key={k++} />);
    } else {
      out.push(
        <span key={k++}>{renderInline(line, sources, onCitation)}</span>,
      );
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
  onFeedback?: (messageId: string, value: FeedbackValue) => void;
}

export function MessageItem({
  message,
  isStreaming,
  onRetry,
  onFeedback,
}: MessageItemProps) {
  // 同一时刻只展开一张来源卡;null 表示全部收起
  const [openSourceN, setOpenSourceN] = useState<number | null>(null);

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
  const sources = message.sources ?? NO_SOURCES;
  const openSource =
    openSourceN === null
      ? undefined
      : sources.find((s) => s.n === openSourceN);
  const feedbackLocked = message.feedback !== undefined;

  const feedbackButtonClass = (selected: boolean) =>
    [
      "rounded-full px-2 py-0.5 text-sm transition-colors",
      selected ? "bg-accent text-accent-ink" : "bg-bg text-text-dim",
      feedbackLocked ? "cursor-default" : "hover:bg-border",
      feedbackLocked && !selected ? "opacity-50" : "",
    ].join(" ");

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] rounded-2xl bg-bubble-bot px-3.5 py-2.5 text-sm leading-relaxed text-text">
        {message.tool && (
          <div className="mb-1.5 flex w-fit items-center gap-1 rounded-full bg-bg px-2 py-0.5 text-xs text-accent">
            <span aria-hidden="true">🔧</span>
            <span>{message.tool.label}</span>
          </div>
        )}
        {renderMarkdown(message.content, sources, (n) =>
          setOpenSourceN((cur) => (cur === n ? null : n)),
        )}
        {isStreaming && (
          <span className="cursor-blink ml-0.5 inline-block h-4 w-0.5 translate-y-0.5 bg-accent" />
        )}
        {message.aborted && (
          <span className="ml-2 text-xs text-text-dim">已停止</span>
        )}
        {openSource && (
          <div className="mt-2 rounded-xl border border-border bg-bg/80 px-3 py-2 text-xs leading-relaxed">
            <div className="mb-1 flex items-start justify-between gap-2">
              <span className="break-all font-medium text-accent">
                [{openSource.n}] {openSource.section_path || "未知章节"}
              </span>
              <button
                type="button"
                onClick={() => setOpenSourceN(null)}
                aria-label="关闭来源"
                className="shrink-0 text-text-dim transition-colors hover:text-text"
              >
                ×
              </button>
            </div>
            <p className="whitespace-pre-wrap break-words text-text-dim">
              {openSource.answer}
            </p>
          </div>
        )}
        {!isStreaming && message.content.trim() !== "" && (
          <div className="mt-2 flex items-center gap-2 border-t border-border/60 pt-1.5">
            <button
              type="button"
              disabled={feedbackLocked}
              aria-pressed={message.feedback === "up"}
              aria-label="有帮助"
              onClick={() => onFeedback?.(message.id, "up")}
              className={feedbackButtonClass(message.feedback === "up")}
            >
              👍
            </button>
            <button
              type="button"
              disabled={feedbackLocked}
              aria-pressed={message.feedback === "down"}
              aria-label="没帮助"
              onClick={() => onFeedback?.(message.id, "down")}
              className={feedbackButtonClass(message.feedback === "down")}
            >
              👎
            </button>
            {feedbackLocked && (
              <span className="text-xs text-text-dim">已反馈</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
