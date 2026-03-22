import {
  Bell,
  RefreshCw,
  Search,
  SlidersHorizontal,
  Sidebar as SidebarIcon,
} from "lucide-react";

import type { DashboardPage } from "../navigation";

export function TopBar({
  page,
  onRefresh,
  onToggleSidebar,
}: {
  page: DashboardPage;
  onRefresh: () => void;
  onToggleSidebar: () => void;
}) {
  return (
    <div className="dashboard-topbar">
      <div className="flex min-w-0 items-center gap-2">
        <button type="button" aria-label="Open navigation" className="dashboard-utility-button lg:hidden" onClick={onToggleSidebar}>
          <SidebarIcon className="size-[16px]" />
        </button>
        <div className="flex min-w-0 items-center gap-1.5 text-[12px]">
          <span className="truncate text-[11px] text-[var(--text-tertiary)]">{page.groupTitle}</span>
          <span className="text-[var(--text-faint)]">/</span>
          <span className="truncate text-[13px] font-semibold text-[var(--foreground)]">{page.label}</span>
        </div>
      </div>

      <div className="flex flex-1 items-center justify-end gap-2">
        <label className="dashboard-search-capsule hidden min-[840px]:flex">
          <Search className="size-4 text-[var(--text-tertiary)]" />
          <input
            type="search"
            aria-label={`Search ${page.label}`}
            className="dashboard-search-input"
            placeholder="Search view"
          />
          <span className="dashboard-search-shortcut">/</span>
        </label>

        <button type="button" aria-label="Refresh data" className="dashboard-utility-button" onClick={onRefresh}>
          <RefreshCw className="size-[15px]" />
        </button>
        <button type="button" aria-label="Open notifications" className="dashboard-utility-button">
          <Bell className="size-[15px]" />
        </button>
        <button type="button" aria-label="Open settings" className="dashboard-utility-button">
          <SlidersHorizontal className="size-[15px]" />
        </button>
      </div>
    </div>
  );
}
