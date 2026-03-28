import { Bot, X } from "lucide-react";

import { dashboardNavSections, type DashboardPageId } from "../navigation";

export function Sidebar({
  activePage,
  isMobileOpen,
  onCloseMobile,
  onNavigate,
}: {
  activePage: DashboardPageId;
  isMobileOpen: boolean;
  onCloseMobile: () => void;
  onNavigate: (pageId: DashboardPageId) => void;
}) {
  return (
    <aside className="dashboard-sidebar" data-open={isMobileOpen ? "true" : "false"}>
      <div className="relative px-5 pb-5 pt-6 lg:px-6 lg:pb-6">
        <div className="flex items-center gap-2.5">
          <div className="dashboard-sidebar-avatar">
            <span className="text-[13px] font-semibold text-[var(--foreground)]">A</span>
          </div>
          <div className="min-w-0">
            <div className="truncate text-[14px] font-semibold tracking-[-0.03em] text-[var(--foreground)]">AURA Sys</div>
          </div>
        </div>
        <button type="button" aria-label="Close navigation" onClick={onCloseMobile} className="dashboard-utility-button absolute right-4 top-4 lg:hidden">
          <X className="size-[16px]" />
        </button>
      </div>

      <nav className="dashboard-scroll flex-1 overflow-x-hidden overflow-y-auto px-4 pb-6 lg:px-4 lg:pb-6">
        <div className="space-y-6">
          {dashboardNavSections.map((section) => (
            <div key={section.title}>
              <div className="mb-2 px-2">
                <span className="dashboard-sidebar-section">{section.title}</span>
              </div>
              <div className="space-y-0.5">
                {section.items.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    aria-label={item.label}
                    onClick={() => onNavigate(item.id)}
                    aria-current={activePage === item.id ? "page" : undefined}
                    className={`dashboard-sidebar-item ${
                      activePage === item.id
                        ? "dashboard-sidebar-item--active"
                        : "dashboard-sidebar-item--idle"
                    }`}
                  >
                    <div
                      className={`flex size-7 items-center justify-center rounded-[10px] ${
                        activePage === item.id
                          ? "text-[var(--foreground)]"
                          : "text-[var(--text-tertiary)]"
                      }`}
                    >
                      <item.icon className="size-[18px]" strokeWidth={activePage === item.id ? 1.9 : 1.6} />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className={`truncate text-[13px] ${activePage === item.id ? "font-semibold tracking-[-0.01em]" : "font-normal"}`}>{item.label}</div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </nav>

      <div className="mt-auto border-t border-[rgba(var(--ink-rgb),0.06)] px-6 py-5">
        <div className="flex items-center justify-center gap-2 text-[var(--text-tertiary)]">
          <Bot className="size-4" />
          <span className="text-[12px] font-semibold tracking-[0.18em]">AURA UI</span>
        </div>
      </div>
    </aside>
  );
}
