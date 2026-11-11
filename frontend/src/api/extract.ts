import type { ExtractRequest, ExtractResult } from "../types";

const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  "http://127.0.0.1:8000";

export async function extractAfterSale(text: string): Promise<ExtractResult> {
  const body: ExtractRequest = { text };
  const res = await fetch(`${API_BASE_URL}/extract`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    // 502 等失败:抛错,由调用方静默降级(不显示卡片)
    throw new Error(`extract failed: ${res.status}`);
  }
  return (await res.json()) as ExtractResult;
}
