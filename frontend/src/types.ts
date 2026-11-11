// 与后端契约一一对应的类型

export interface ChatRequest {
  session_id: string;
  message: string;
}

export interface ExtractRequest {
  text: string;
}

export interface ExtractResult {
  order_id: string | null;
  appeal_type: string; // 固定枚举:退款|退货退款|换货|仅退款|补发|维修|物流催单|投诉|其他
  expected_solution: string | null;
}

export type ChatStatus =
  | "idle"
  | "sending"
  | "streaming"
  | "done"
  | "error"
  | "aborted";

export type ChatErrorKind = "network" | "http" | "timeout";

export type Message =
  | { id: string; role: "user"; content: string }
  | { id: string; role: "assistant"; content: string; aborted?: boolean }
  | { id: string; role: "system"; content: string; retryable?: boolean }
  | { id: string; role: "extract"; card: ExtractResult };
