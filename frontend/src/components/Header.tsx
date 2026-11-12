interface HeaderProps {
  onClose?: () => void;
}

export function Header({ onClose }: HeaderProps) {
  return (
    <div className="flex items-center justify-between border-b border-[0.5px] border-border px-4 py-3">
      <div className="flex items-center gap-2.5">
        <div className="relative">
          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-accent text-base font-medium text-accent-ink">
            灯
          </div>
          <span className="absolute -bottom-0.5 -right-0.5 h-2.5 w-2.5 rounded-full border-2 border-bg bg-online" />
        </div>
        <div>
          <div className="text-sm font-medium leading-tight text-text">
            灯下客服
          </div>
          <div className="flex items-center gap-1 text-xs text-online">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-online" />
            在线
          </div>
        </div>
      </div>
      {onClose && (
        <button
          onClick={onClose}
          aria-label="关闭"
          className="rounded-lg p-1.5 text-text-dim transition-colors hover:bg-surface hover:text-text"
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="none"
            aria-hidden="true"
          >
            <path
              d="M4 4l8 8M12 4l-8 8"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
          </svg>
        </button>
      )}
    </div>
  );
}
