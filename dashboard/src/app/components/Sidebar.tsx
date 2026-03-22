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
      <div className="relative px-4 pb-3 pt-4 lg:px-5 lg:pb-4 lg:pt-5">
        <div className="rounded-[22px] border border-[rgba(var(--ink-rgb),0.06)] bg-[rgba(var(--paper-rgb),0.92)] px-3 py-3 shadow-[0_12px_32px_rgba(15,23,42,0.04)]">
          <div className="flex items-center gap-3">
            <div className="inline-flex size-8 shrink-0 items-center justify-center rounded-[12px] border border-[rgba(237,162,87,0.18)] bg-[linear-gradient(135deg,#f4b36e_0%,#e28d4b_100%)] text-white shadow-[0_10px_24px_rgba(226,141,75,0.28)]">
              <Bot className="size-4" />
            </div>
            <div>
              <div className="text-[15px] font-medium tracking-[-0.04em] text-[var(--foreground)]">AURA Sys</div>
              <div className="text-[11px] text-[var(--text-tertiary)]">runtime operations</div>
            </div>
          </div>
        </div>
        <button type="button" aria-label="Close navigation" onClick={onCloseMobile} className="dashboard-utility-button absolute right-4 top-4 lg:hidden">
          <X className="size-[16px]" />
        </button>
      </div>

      <nav className="dashboard-scroll flex-1 overflow-x-hidden overflow-y-auto px-4 pb-4 lg:pb-6">
        <div className="space-y-5">
          {dashboardNavSections.map((section) => (
            <div key={section.title}>
              <div className="mb-2 px-3">
                <span className="dashboard-eyebrow">{section.title}</span>
              </div>
              <div className="space-y-1">
                {section.items.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    aria-label={item.label}
                    onClick={() => onNavigate(item.id)}
                    aria-current={activePage === item.id ? "page" : undefined}
                    className={`relative flex w-full items-center gap-3 rounded-[16px] px-3 py-2.5 text-left transition-colors ${
                      activePage === item.id
                        ? "border border-[rgba(var(--ink-rgb),0.06)] bg-[rgba(var(--ink-rgb),0.04)] text-[var(--foreground)] shadow-[0_10px_24px_rgba(15,23,42,0.04)]"
                        : "text-[var(--text-secondary)] hover:bg-[rgba(var(--ink-rgb),0.03)]"
                    }`}
                  >
                    <div
                      className={`flex size-8 items-center justify-center rounded-[12px] ${
                        activePage === item.id
                          ? "bg-[rgba(var(--ink-rgb),0.05)] text-[var(--foreground)]"
                          : "bg-transparent text-[var(--text-tertiary)]"
                      }`}
                    >
                      <item.icon className="size-[18px]" strokeWidth={activePage === item.id ? 1.9 : 1.6} />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className={`truncate text-[13px] ${activePage === item.id ? "font-medium tracking-[-0.02em]" : "font-normal"}`}>{item.label}</div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </nav>

    </aside>
  );
}
