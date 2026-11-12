import type { ChatErrorKind, ChatRequest } from "../types";

const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  "http://127.0.0.1:8000";

const NO_DATA_TIMEOUT_MS = 30_000;

export interface StreamHandlers {
  onToken: (token: string) => void;
  onErrorFrame: (message: string) => void;
  onError: (kind: ChatErrorKind) => void;
  onDone: () => void;
  onTool: (tool: { name: string; label: string }) => void;
}

/**
 * 按 SSE 规范手动解析字节流(不用 EventSource,因为要 POST)。
 * 每收到一个事件,把该事件的所有 data: 行用 \n 拼接后回调。
 */
async function parseSSE(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  onData: (data: string) => void,
): Promise<void> {
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let dataLines: string[] = [];

  const emit = () => {
    if (dataLines.length > 0) {
      const data = dataLines.join("\n");
      dataLines = [];
      onData(data);
    }
  };

  const handleLine = (line: string) => {
    if (line === "") {
      emit();
    } else if (line.startsWith("data:")) {
      // 去掉 "data:" 前缀与可选的一个空格
      dataLines.push(line.slice(5).replace(/^ /, ""));
    }
    // 其它字段(event:/id:/retry:/注释)忽略
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, idx).replace(/\r$/, "");
      buffer = buffer.slice(idx + 1);
      handleLine(line);
    }
  }
  buffer += decoder.decode();
  if (buffer) handleLine(buffer);
  emit();
}

export async function streamChat(
  params: ChatRequest,
  handlers: StreamHandlers,
  signal: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: params.session_id,
        message: params.message,
      }),
      signal,
    });
  } catch {
    if (signal.aborted) return; // 用户主动停止
    handlers.onError("network");
    return;
  }

  if (!res.ok) {
    handlers.onError("http");
    return;
  }

  if (!res.body) {
    handlers.onError("network");
    return;
  }

  const reader = res.body.getReader();
  let timedOut = false;

  // 30 秒无任何数据帧即判定超时;用 reader.cancel() 打断阻塞中的 read()
  let timer = setTimeout(() => {
    timedOut = true;
    void reader.cancel();
  }, NO_DATA_TIMEOUT_MS);
  const resetTimer = () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      timedOut = true;
      void reader.cancel();
    }, NO_DATA_TIMEOUT_MS);
  };

  try {
    await parseSSE(reader, (data) => {
      if (data === "[DONE]") {
        clearTimeout(timer);
        handlers.onDone();
        return;
      }
      resetTimer();
      // 后端 error 帧是唯一带 JSON 的 data
      if (data.startsWith("{")) {
        try {
          const obj: unknown = JSON.parse(data);
          if (
            obj &&
            typeof obj === "object" &&
            "type" in obj &&
            (obj as { type: unknown }).type === "tool"
          ) {
            const tool = obj as { tool?: unknown; label?: unknown };
            handlers.onTool({
              name: typeof tool.tool === "string" ? tool.tool : "",
              label: typeof tool.label === "string" ? tool.label : "",
            });
            return;
          }
          if (
            obj &&
            typeof obj === "object" &&
            "error" in obj &&
            typeof (obj as { error: unknown }).error === "string"
          ) {
            handlers.onErrorFrame((obj as { error: string }).error);
            return;
          }
        } catch {
          // 非法 JSON,当普通 token
        }
      }
      handlers.onToken(data);
    });
    // 读取正常结束([DONE] 之后后端关闭连接)
    clearTimeout(timer);
    if (timedOut) handlers.onError("timeout");
  } catch {
    clearTimeout(timer);
    if (signal.aborted) return; // 用户主动停止
    if (timedOut) handlers.onError("timeout");
    else handlers.onError("network");
  }
}
