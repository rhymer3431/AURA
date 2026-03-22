import { Bot } from "lucide-react";

import { dashboardNavSections, type DashboardPageId } from "../navigation";

export function Sidebar({
  activePage,
  onNavigate,
}: {
  activePage: DashboardPageId;
  onNavigate: (pageId: DashboardPageId) => void;
}) {
  return (
    <aside className="dashboard-sidebar">
      <div className="px-4 pt-4 pb-3 lg:px-5 lg:pt-5 lg:pb-4">
        <div className="dashboard-panel-strong rounded-[24px] px-4 py-4">
          <div className="flex items-center gap-3">
            <div className="inline-flex size-10 shrink-0 items-center justify-center rounded-full border border-[rgba(24,33,37,0.08)] bg-[rgba(24,33,37,0.96)] text-white">
              <Bot className="size-4" />
            </div>
            <div>
              <div className="text-[15px] font-semibold tracking-[-0.03em] text-[var(--foreground)]">AURA Sys</div>
              <div className="dashboard-micro mt-1">runtime diagnostics console</div>
            </div>
          </div>
        </div>
      </div>

      <nav className="dashboard-scroll flex-1 overflow-x-auto overflow-y-hidden px-4 pb-4 lg:overflow-y-auto lg:overflow-x-hidden lg:pb-6">
        <div className="flex gap-3 lg:block lg:space-y-6">
        {dashboardNavSections.map((section) => (
          <div key={section.title} className="shrink-0 min-w-[220px] lg:min-w-0">
            <div className="px-3 mb-2">
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
                  className={`relative w-full flex items-center gap-3 px-3 py-2.5 rounded-[18px] cursor-pointer transition-colors text-left ${
                    activePage === item.id
                      ? "bg-[rgba(255,252,246,0.88)] border border-[rgba(79,152,168,0.15)] text-[var(--foreground)]"
                      : "text-[var(--text-secondary)] hover:bg-[rgba(24,33,37,0.035)]"
                  }`}
                >
                  {activePage === item.id ? (
                    <span className="absolute bottom-2 left-1 top-2 w-[3px] rounded-full bg-[var(--signal-cyan)]" />
                  ) : null}
                  <div
                    className={`flex size-8 items-center justify-center rounded-full ${
                      activePage === item.id
                        ? "bg-[rgba(79,152,168,0.12)] text-[var(--signal-cyan)]"
                        : "bg-transparent text-[var(--text-tertiary)]"
                    }`}
                  >
                    <item.icon className="size-5" strokeWidth={activePage === item.id ? 2 : 1.5} />
                  </div>
                  <div className="min-w-0">
                    <div className={`text-[13px] ${activePage === item.id ? "font-semibold tracking-[-0.02em]" : "font-medium"}`}>{item.label}</div>
                    <div className="mt-0.5 line-clamp-1 text-[11px] text-[var(--text-tertiary)]">{item.groupTitle}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        ))}
        </div>
      </nav>

      <div className="mt-auto hidden border-t border-[rgba(24,33,37,0.06)] p-5 lg:block">
        <div className="dashboard-panel-strong rounded-[20px] px-4 py-3">
          <div className="dashboard-eyebrow mb-2">Telemetry Rail</div>
          <div className="flex items-center gap-2 text-[12px] text-[var(--text-secondary)]">
            <Bot className="size-4 text-[var(--signal-cyan)]" />
            <span>AURA UI</span>
            <span className="ml-auto dashboard-micro">ops</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
