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
      <div className="relative px-4 pb-4 pt-5 lg:px-5 lg:pb-5 lg:pt-6">
        <div className="flex items-center gap-3">
          <div className="inline-flex size-10 shrink-0 items-center justify-center rounded-[14px] bg-[var(--tone-violet-bg)] text-[var(--foreground)]">
            <Bot className="size-4" />
          </div>
          <div className="min-w-0">
            <div className="truncate text-[15px] font-semibold tracking-[-0.03em] text-[var(--foreground)]">AURA</div>
            <div className="truncate text-[12px] text-[var(--text-tertiary)]">runtime dashboard</div>
          </div>
        </div>
        <button type="button" aria-label="Close navigation" onClick={onCloseMobile} className="dashboard-utility-button absolute right-4 top-4 lg:hidden">
          <X className="size-[16px]" />
        </button>
      </div>

      <nav className="dashboard-scroll flex-1 overflow-x-hidden overflow-y-auto px-4 pb-4 lg:pb-6">
        <div className="space-y-6">
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
                        ? "bg-[rgba(var(--ink-rgb),0.04)] text-[var(--foreground)]"
                        : "text-[var(--text-secondary)] hover:bg-[rgba(var(--ink-rgb),0.025)]"
                    }`}
                  >
                    <div
                      className={`flex size-8 items-center justify-center rounded-[12px] ${
                        activePage === item.id
                          ? "bg-white text-[var(--foreground)]"
                          : "bg-transparent text-[var(--text-tertiary)]"
                      }`}
                    >
                      <item.icon className="size-[18px]" strokeWidth={activePage === item.id ? 1.9 : 1.6} />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className={`truncate text-[14px] ${activePage === item.id ? "font-semibold tracking-[-0.01em]" : "font-normal"}`}>{item.label}</div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </nav>

      <div className="px-5 pb-5 pt-3">
        <div className="rounded-[18px] bg-[var(--surface-2)] px-4 py-3">
          <div className="dashboard-eyebrow mb-1">Workspace</div>
          <div className="text-[13px] font-semibold text-[var(--foreground)]">SnowUI control room</div>
          <div className="mt-1 text-[12px] text-[var(--text-tertiary)]">Minimal layout, live runtime context.</div>
        </div>
      </div>
    </aside>
  );
}
