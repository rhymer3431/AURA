import {
  Sidebar as SidebarIcon,
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
      <div className="flex min-w-0 items-center gap-2.5">
        <button type="button" aria-label="Open navigation" className="dashboard-utility-button lg:hidden" onClick={onToggleSidebar}>
          <SidebarIcon className="size-[18px]" />
        </button>
        <div className="flex min-w-0 items-center gap-2 text-[13px]">
          <span className="truncate text-[12px] text-[var(--text-secondary)]">{page.groupTitle}</span>
          <span className="text-[var(--text-faint)]">/</span>
          <span className="truncate font-medium text-[var(--foreground)]">{page.label}</span>
        </div>
      </div>
    </div>
  );
}
