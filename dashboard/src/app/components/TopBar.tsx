import {
  Sidebar as SidebarIcon,
  Star,
} from "lucide-react";

import type { DashboardPage } from "../navigation";

export function TopBar({
  page,
  onToggleSidebar,
}: {
  page: DashboardPage;
  onToggleSidebar: () => void;
}) {
  return (
    <div className="dashboard-topbar">
      <div className="flex min-w-0 items-center gap-3">
        <button type="button" aria-label="Open navigation" className="dashboard-utility-button lg:hidden" onClick={onToggleSidebar}>
          <SidebarIcon className="size-[16px]" />
        </button>
        <div className="hidden lg:flex dashboard-topbar-icon">
          <SidebarIcon className="size-[16px]" />
        </div>
        <button type="button" aria-label="Favorite page" className="dashboard-topbar-icon">
          <Star className="size-[16px]" />
        </button>
        <div className="flex min-w-0 items-center gap-2 text-[13px]">
          <span className="truncate text-[var(--text-tertiary)]">{page.groupTitle}</span>
          <span className="text-[var(--text-faint)]">/</span>
          <span className="truncate font-medium text-[var(--foreground)]">{page.label}</span>
        </div>
      </div>
    </div>
  );
}
