import { useCallback, useReducer, useRef } from "react";
import { streamChat } from "../api/chat";
import { extractAfterSale } from "../api/extract";
import type {
  ChatErrorKind,
  ChatStatus,
  ExtractResult,
  Message,
} from "../types";

const SESSION_KEY = "lamp-cs-session-id";

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
  | { type: "stream_start" }
  | { type: "token"; text: string }
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
          { id: crypto.randomUUID(), role: "assistant", content: "" },
        ],
      };
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
            dispatch({ type: "stream_start" });
          }
          dispatch({ type: "token", text: t });
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
    hasUserMessage,
  };
}
