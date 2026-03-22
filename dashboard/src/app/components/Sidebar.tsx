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
      <div className="relative px-3.5 pb-3.5 pt-4 lg:px-4 lg:pb-4 lg:pt-5">
        <div className="flex items-center gap-2.5">
          <div className="inline-flex size-9 shrink-0 items-center justify-center rounded-[12px] bg-[var(--tone-violet-bg)] text-[var(--foreground)]">
            <Bot className="size-4" />
          </div>
          <div className="min-w-0">
            <div className="truncate text-[14px] font-semibold tracking-[-0.03em] text-[var(--foreground)]">AURA</div>
            <div className="truncate text-[11px] text-[var(--text-tertiary)]">runtime dashboard</div>
          </div>
        </div>
        <button type="button" aria-label="Close navigation" onClick={onCloseMobile} className="dashboard-utility-button absolute right-3.5 top-3.5 lg:hidden">
          <X className="size-[16px]" />
        </button>
      </div>

      <nav className="dashboard-scroll flex-1 overflow-x-hidden overflow-y-auto px-3.5 pb-3.5 lg:pb-5">
        <div className="space-y-5">
          {dashboardNavSections.map((section) => (
            <div key={section.title}>
              <div className="mb-1.5 px-2.5">
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
                    className={`relative flex w-full items-center gap-2.5 rounded-[14px] px-2.5 py-2 text-left transition-colors ${
                      activePage === item.id
                        ? "bg-[rgba(var(--ink-rgb),0.04)] text-[var(--foreground)]"
                        : "text-[var(--text-secondary)] hover:bg-[rgba(var(--ink-rgb),0.025)]"
                    }`}
                  >
                    <div
                      className={`flex size-7 items-center justify-center rounded-[10px] ${
                        activePage === item.id
                          ? "bg-white text-[var(--foreground)]"
                          : "bg-transparent text-[var(--text-tertiary)]"
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

      <div className="px-4 pb-4 pt-2.5">
        <div className="rounded-[16px] border border-[rgba(var(--ink-rgb),0.06)] bg-[var(--surface-2)] px-3.5 py-3">
          <div className="dashboard-eyebrow mb-1">Workspace</div>
          <div className="text-[12px] font-semibold text-[var(--foreground)]">SnowUI control room</div>
          <div className="mt-1 text-[11px] text-[var(--text-tertiary)]">Minimal layout, live runtime context.</div>
        </div>
      </div>
    </aside>
  );
}
