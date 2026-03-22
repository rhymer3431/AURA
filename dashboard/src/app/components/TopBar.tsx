import {
  Sidebar as SidebarIcon,
  Star,
  Radio,
} from "lucide-react";

import { useDashboard } from "../state";
import type { DashboardPage } from "../navigation";
import { ConsoleBadge } from "./console-ui";

export function TopBar({ page }: { page: DashboardPage }) {
  const { state } = useDashboard();

  return (
    <div className="dashboard-topbar">
      <div className="flex items-center gap-4">
        <button className="dashboard-button-secondary !px-3 !py-2 text-[var(--text-secondary)]">
          <SidebarIcon className="size-[18px]" />
        </button>
        <button className="dashboard-button-secondary !px-3 !py-2 text-[var(--text-secondary)]">
          <Star className="size-[18px]" />
        </button>
        <div className="flex items-center gap-2 text-[13px]">
          <span className="dashboard-micro">{page.groupTitle}</span>
          <span className="text-[var(--text-faint)]">/</span>
          <span className="font-medium text-[var(--foreground)]">{page.label}</span>
        </div>
      </div>

      <div className="flex items-center gap-2 text-[12px]">
        <ConsoleBadge tone={state?.session.active ? "emerald" : "amber"}>
          <Radio className="size-3" />
          {state?.session.active ? "session running" : "session idle"}
        </ConsoleBadge>
        <ConsoleBadge tone="slate" dot={false}>
          exec {String(state?.runtime.executionMode ?? state?.runtime.modes?.executionMode ?? "IDLE")}
        </ConsoleBadge>
      </div>
    </div>
  );
}
