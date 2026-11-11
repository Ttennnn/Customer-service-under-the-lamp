const QUICK_REPLIES = ["我要退款", "物流一直没更新", "怎么换货"];

interface QuickRepliesProps {
  onPick: (text: string) => void;
}

export function QuickReplies({ onPick }: QuickRepliesProps) {
  return (
    <div className="flex flex-wrap gap-2">
      {QUICK_REPLIES.map((q) => (
        <button
          key={q}
          onClick={() => onPick(q)}
          className="rounded-full border border-[0.5px] border-border bg-surface px-3 py-1.5 text-sm text-text transition-colors hover:border-accent hover:text-accent"
        >
          {q}
        </button>
      ))}
    </div>
  );
}
