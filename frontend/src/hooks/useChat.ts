import { useCallback, useReducer, useRef } from "react";
import { streamChat } from "../api/chat";
import { extractAfterSale } from "../api/extract";
import type {
  ChatErrorKind,
  ChatStatus,
  CitationSource,
  ExtractResult,
  FeedbackValue,
  Message,
} from "../types";

const SESSION_KEY = "lamp-cs-session-id";
const FEEDBACK_KEY = "lamp-cs-feedback";

interface FeedbackRecord {
  sessionId: string;
  messageId: string;
  value: FeedbackValue;
  content: string;
  createdAt: string;
}

/**
 * 追加一条反馈记录。按 messageId 幂等:StrictMode 下 reducer 可能被重复执行,
 * 再次调用不应写出第二条。
 */
function appendFeedbackRecord(record: FeedbackRecord): void {
  try {
    const raw = localStorage.getItem(FEEDBACK_KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : [];
    const list: unknown[] = Array.isArray(parsed) ? parsed : [];
    const exists = list.some(
      (item) =>
        !!item &&
        typeof item === "object" &&
        (item as { messageId?: unknown }).messageId === record.messageId,
    );
    if (exists) return;
    list.push(record);
    localStorage.setItem(FEEDBACK_KEY, JSON.stringify(list));
  } catch {
    // localStorage 不可用(隐私模式/配额满)时不阻断交互
  }
}

function getOrCreateSessionId(): string {
  const existing = sessionStorage.getItem(SESSION_KEY);
  if (existing) return existing;
  const id = crypto.randomUUID();
  sessionStorage.setItem(SESSION_KEY, id);
  return id;
}

const ERROR_TEXT: Record<ChatErrorKind, string> = {
  network: "网络连接中断,请检查网络后重试。",
  http: "服务暂时不可用,请稍后重试。",
  timeout: "响应超时,请稍后重试。",
};

interface ChatState {
  sessionId: string;
  messages: Message[];
  status: ChatStatus;
  errorKind: ChatErrorKind | null;
}

type Action =
  | { type: "send"; message: string }
  | { type: "resend" }
  | {
      type: "stream_start";
      tool?: { name: string; label: string };
      sources?: CitationSource[];
    }
  | { type: "token"; text: string }
  | { type: "feedback"; messageId: string; value: FeedbackValue }
  | { type: "error_frame"; text: string }
  | { type: "fail"; kind: ChatErrorKind }
  | { type: "done" }
  | { type: "abort" }
  | { type: "extract_card"; card: ExtractResult };

function reducer(state: ChatState, action: Action): ChatState {
  switch (action.type) {
    case "send":
      return {
        ...state,
        status: "sending",
        errorKind: null,
        messages: [
          ...state.messages,
          { id: crypto.randomUUID(), role: "user", content: action.message },
        ],
      };
    case "resend": {
      // 去掉末尾可重试的错误提示,重新发起同一消息
      const messages = [...state.messages];
      let last = messages[messages.length - 1];
      while (last && last.role === "system" && last.retryable) {
        messages.pop();
        last = messages[messages.length - 1];
      }
      return { ...state, messages, status: "sending", errorKind: null };
    }
    case "stream_start":
      return {
        ...state,
        status: "streaming",
        messages: [
          ...state.messages,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: "",
            ...(action.tool ? { tool: action.tool } : {}),
            ...(action.sources ? { sources: action.sources } : {}),
          },
        ],
      };
    case "feedback": {
      const idx = state.messages.findIndex((m) => m.id === action.messageId);
      if (idx < 0) return state;
      const target = state.messages[idx];
      // 一次性锁定:已有反馈的消息不再改变,也不写第二条记录
      if (target.role !== "assistant" || target.feedback) return state;
      appendFeedbackRecord({
        sessionId: state.sessionId,
        messageId: action.messageId,
        value: action.value,
        content: target.content,
        createdAt: new Date().toISOString(),
      });
      const messages = [...state.messages];
      messages[idx] = { ...target, feedback: action.value };
      return { ...state, messages };
    }
    case "token": {
      const messages = [...state.messages];
      const last = messages[messages.length - 1];
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = {
          ...last,
          content: last.content + action.text,
        };
      }
      return { ...state, messages };
    }
    case "error_frame":
      return {
        ...state,
        messages: [
          ...state.messages,
          { id: crypto.randomUUID(), role: "system", content: action.text },
        ],
      };
    case "fail":
      return {
        ...state,
        status: "error",
        errorKind: action.kind,
        messages: [
          ...state.messages,
          {
            id: crypto.randomUUID(),
            role: "system",
            content: ERROR_TEXT[action.kind],
            retryable: true,
          },
        ],
      };
    case "done":
      return { ...state, status: "done" };
    case "abort": {
      const messages = [...state.messages];
      const last = messages[messages.length - 1];
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, aborted: true };
      }
      return { ...state, messages, status: "aborted" };
    }
    case "extract_card":
      return {
        ...state,
        messages: [
          ...state.messages,
          { id: crypto.randomUUID(), role: "extract", card: action.card },
        ],
      };
    default:
      return state;
  }
}

export function useChat() {
  const sessionIdRef = useRef<string>(getOrCreateSessionId());
  const abortRef = useRef<AbortController | null>(null);
  const pendingToolRef = useRef<{ name: string; label: string } | null>(null);
  const pendingSourcesRef = useRef<CitationSource[] | null>(null);
  const [state, dispatch] = useReducer(reducer, {
    sessionId: sessionIdRef.current,
    messages: [],
    status: "idle",
    errorKind: null,
  });

  const startStream = useCallback((text: string, appendUser: boolean) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    // 新一轮开始,清掉上一轮未消费的 sources,避免串到新回答
    pendingSourcesRef.current = null;

    if (appendUser) {
      dispatch({ type: "send", message: text });
    } else {
      dispatch({ type: "resend" });
    }

    let started = false;
    void streamChat(
      { session_id: sessionIdRef.current, message: text },
      {
        onToken: (t) => {
          if (!started) {
            started = true;
            const tool = pendingToolRef.current;
            pendingToolRef.current = null;
            const sources = pendingSourcesRef.current;
            pendingSourcesRef.current = null;
            dispatch({
              type: "stream_start",
              tool: tool ?? undefined,
              sources: sources ?? undefined,
            });
          }
          dispatch({ type: "token", text: t });
        },
        onTool: (tool) => {
          pendingToolRef.current = tool;
        },
        onSources: (sources) => {
          pendingSourcesRef.current = sources;
        },
        onErrorFrame: (t) => dispatch({ type: "error_frame", text: t }),
        onError: (kind) => dispatch({ type: "fail", kind }),
        onDone: () => dispatch({ type: "done" }),
      },
      controller.signal,
    );
  }, []);

  const send = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      startStream(trimmed, true);
    },
    [startStream],
  );

  const retry = useCallback(() => {
    const lastUser = [...state.messages]
      .reverse()
      .find((m) => m.role === "user");
    if (!lastUser) return;
    startStream(lastUser.content, false);
  }, [startStream, state.messages]);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    dispatch({ type: "abort" });
  }, []);

  const onFeedback = useCallback((messageId: string, value: FeedbackValue) => {
    dispatch({ type: "feedback", messageId, value });
  }, []);

  const extract = useCallback(async () => {
    const lastUser = [...state.messages]
      .reverse()
      .find((m) => m.role === "user");
    if (!lastUser) return;
    try {
      const card = await extractAfterSale(lastUser.content);
      dispatch({ type: "extract_card", card });
    } catch {
      // 502 / 网络失败:静默降级,不显示卡片
    }
  }, [state.messages]);

  const hasUserMessage = state.messages.some((m) => m.role === "user");

  return {
    messages: state.messages,
    status: state.status,
    sessionId: state.sessionId,
    send,
    retry,
    stop,
    extract,
    onFeedback,
    hasUserMessage,
  };
}
