import { Widget } from "./components/Widget";

export default function App() {
  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden p-4">
      {/* 周围点缀:暖色光点 */}
      <div aria-hidden className="pointer-events-none absolute inset-0">
        <span className="absolute left-[10%] top-[16%] h-2 w-2 rounded-full bg-accent/40" />
        <span className="absolute right-[12%] top-[28%] h-1.5 w-1.5 rounded-full bg-accent/25" />
        <span className="absolute left-[18%] bottom-[18%] h-1.5 w-1.5 rounded-full bg-text-dim/30" />
        <span className="absolute right-[20%] bottom-[14%] h-2 w-2 rounded-full bg-online/30" />
        <span className="absolute left-[42%] top-[7%] h-1 w-1 rounded-full bg-accent/30" />
        <span className="absolute right-[40%] bottom-[7%] h-1 w-1 rounded-full bg-text-dim/25" />
      </div>

      <Widget />
    </div>
  );
}
