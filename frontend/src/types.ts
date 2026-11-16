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

/** 回答引用的来源 chunk,与后端 sources 帧一一对应。 */
export interface CitationSource {
  n: number;
  chunk_id: number;
  section_path: string;
  question: string;
  answer: string;
  category: string;
  content_type: string;
  score: number;
}

export type FeedbackValue = "up" | "down";

export type Message =
  | { id: string; role: "user"; content: string }
  | {
      id: string;
      role: "assistant";
      content: string;
      aborted?: boolean;
      tool?: { name: string; label: string };
      sources?: CitationSource[];
      feedback?: FeedbackValue;
    }
  | { id: string; role: "system"; content: string; retryable?: boolean }
  | { id: string; role: "extract"; card: ExtractResult };
