import {
  Sidebar as SidebarIcon,
  Star,
  Radio,
  Search,
  Sun,
  History,
  Bell,
} from "lucide-react";

import { useDashboard } from "../state";
import type { DashboardPage } from "../navigation";
import { ConsoleBadge } from "./console-ui";

export function TopBar({
  page,
  onToggleSidebar,
}: {
  page: DashboardPage;
  onToggleSidebar: () => void;
}) {
  const { state } = useDashboard();

  return (
    <div className="dashboard-topbar">
      <div className="flex min-w-0 items-center gap-2.5">
        <button type="button" aria-label="Open navigation" className="dashboard-utility-button lg:hidden" onClick={onToggleSidebar}>
          <SidebarIcon className="size-[18px]" />
        </button>
        <button type="button" className="dashboard-utility-button">
          <Star className="size-[16px]" />
        </button>
        <div className="flex min-w-0 items-center gap-2 text-[13px]">
          <span className="truncate text-[12px] text-[var(--text-secondary)]">{page.groupTitle}</span>
          <span className="text-[var(--text-faint)]">/</span>
          <span className="truncate font-medium text-[var(--foreground)]">{page.label}</span>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-end gap-2 text-[12px]">
        <button type="button" className="dashboard-search-capsule hidden xl:inline-flex">
          <Search className="size-3.5" />
          <span className="min-w-0 flex-1 text-left">Search</span>
          <span className="dashboard-search-shortcut">/</span>
        </button>
        <button type="button" className="dashboard-utility-button">
          <Sun className="size-[15px]" />
        </button>
        <button type="button" className="dashboard-utility-button">
          <History className="size-[15px]" />
        </button>
        <button type="button" className="dashboard-utility-button">
          <Bell className="size-[15px]" />
        </button>
        <ConsoleBadge tone={state?.session.active ? "emerald" : "amber"}>
          <Radio className="size-3" />
          {state?.session.active ? "session live" : "session idle"}
        </ConsoleBadge>
        <ConsoleBadge tone="slate" dot={false}>
          exec {String(state?.runtime.executionMode ?? state?.runtime.modes?.executionMode ?? "IDLE")}
        </ConsoleBadge>
      </div>
    </div>
  );
}
