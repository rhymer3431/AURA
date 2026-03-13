import {
  Sidebar as SidebarIcon,
  Star,
  Radio,
} from "lucide-react";

import { useDashboard } from "../state";

export function TopBar() {
  const { state } = useDashboard();

  return (
    <div className="flex items-center justify-between px-8 py-5 bg-white shrink-0">
      <div className="flex items-center gap-4">
        <button className="text-black/40 hover:text-black/80 transition-colors">
          <SidebarIcon className="size-[18px]" />
        </button>
        <button className="text-black/40 hover:text-black/80 transition-colors">
          <Star className="size-[18px]" />
        </button>
        <div className="flex items-center gap-2 text-[14px]">
          <span className="text-black/40">Dashboards</span>
          <span className="text-black/20">/</span>
          <span className="text-black/90">Pipeline Overview</span>
        </div>
      </div>

      <div className="flex items-center gap-3 text-[12px] text-black/55">
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-black/[0.03]">
          <Radio className={`size-3 ${state?.session.active ? "text-emerald-500" : "text-amber-500"}`} />
          {state?.session.active ? "session running" : "session idle"}
        </span>
        <span>
          mode: <span className="text-black/80 font-medium">{state?.session.config?.plannerMode ?? "none"}</span>
        </span>
      </div>
    </div>
  );
}
