import type { ExtractResult } from "../types";

const APPEAL_COLORS: Record<string, string> = {
  "退款": "#F0B76A",
  "退货退款": "#C97B5A",
  "换货": "#8FA98C",
  "仅退款": "#D9B26F",
  "补发": "#7FA3B0",
  "维修": "#9A8CB0",
  "物流催单": "#B08D57",
  "投诉": "#C96A6A",
  "其他": "#8C8378",
};

function AppealTag({ type }: { type: string }) {
  const color = APPEAL_COLORS[type] ?? "#8C8378";
  return (
    <span
      className="rounded-full px-2 py-0.5 text-xs"
      style={{ backgroundColor: `${color}1F`, color }}
    >
      {type}
    </span>
  );
}

export function ExtractCard({ card }: { card: ExtractResult }) {
  return (
    <div className="flex justify-start">
      <div className="w-[85%] rounded-2xl border border-[0.5px] border-border bg-surface p-3">
        <div className="mb-2 flex items-center justify-between">
          <span className="text-xs text-text-dim">识别诉求</span>
          <AppealTag type={card.appeal_type} />
        </div>
        <div className="space-y-1 text-sm text-text">
          <div className="flex justify-between gap-3">
            <span className="shrink-0 text-text-dim">订单号</span>
            <span className="text-right">{card.order_id ?? "未提供"}</span>
          </div>
          <div className="flex justify-between gap-3">
            <span className="shrink-0 text-text-dim">期望方案</span>
            <span className="text-right">{card.expected_solution ?? "未说明"}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
