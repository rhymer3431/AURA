import { useEffect, useState } from "react";

import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { StatCards } from "./components/StatCards";
import { PipelineFlow } from "./components/PipelineFlow";
import { NavigationControlPanel } from "./components/NavigationControlPanel";
import { OccupancyMapPanel } from "./components/OccupancyMapPanel";
import { ExternalServicesPanel } from "./components/ExternalServicesPanel";
import { RobotViewer } from "./components/RobotViewer";
import { ControlStrip } from "./components/ControlStrip";
import {
  ProcessesWidget,
  SensorsWidget,
  PerceptionWidget,
  MemoryWidget,
  IpcOrchestrationWidget,
  LogsWidget,
} from "./components/SystemStatusWidgets";
import { ExecutionModesPanel } from "./components/ExecutionModesPanel";
import { ArtifactsStoragePanel } from "./components/ArtifactsStoragePanel";
import { DashboardSectionTabs } from "./components/DashboardSectionTabs";
import { DashboardPageHero } from "./components/DashboardPageHero";
import { DashboardActivityRail } from "./components/DashboardActivityRail";
import {
  DEFAULT_DASHBOARD_PAGE,
  dashboardPageHash,
  dashboardPages,
  parseDashboardPageId,
  type DashboardPageId,
} from "./navigation";
import { useDashboard } from "./state";
import { ConsoleBadge } from "./components/console-ui";

function currentPageFromLocation(): DashboardPageId {
  if (typeof window === "undefined") {
    return DEFAULT_DASHBOARD_PAGE;
  }
  return parseDashboardPageId(window.location.hash);
}

export default function App() {
  const { error, state } = useDashboard();
  const [activePage, setActivePage] = useState<DashboardPageId>(() => currentPageFromLocation());
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") {
      return undefined;
    }

    const syncPageFromHash = () => {
      const nextPage = currentPageFromLocation();
      setActivePage(nextPage);
      const nextHash = dashboardPageHash(nextPage);
      if (window.location.hash !== nextHash) {
        window.history.replaceState(null, "", nextHash);
      }
    };

    syncPageFromHash();
    window.addEventListener("hashchange", syncPageFromHash);
    return () => window.removeEventListener("hashchange", syncPageFromHash);
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return undefined;
    }

    const syncDesktopLayout = () => {
      if (window.innerWidth >= 1024) {
        setMobileSidebarOpen(false);
      }
    };

    syncDesktopLayout();
    window.addEventListener("resize", syncDesktopLayout);
    return () => window.removeEventListener("resize", syncDesktopLayout);
  }, []);

  useEffect(() => {
    if (typeof document === "undefined") {
      return undefined;
    }

    const previousOverflow = document.body.style.overflow;
    if (mobileSidebarOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = previousOverflow;
    }

    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [mobileSidebarOpen]);

  const page = dashboardPages[activePage];

  function navigateTo(pageId: DashboardPageId) {
    setMobileSidebarOpen(false);
    if (typeof window === "undefined") {
      setActivePage(pageId);
      return;
    }
    const nextHash = dashboardPageHash(pageId);
    if (window.location.hash === nextHash) {
      setActivePage(pageId);
      return;
    }
    window.location.hash = nextHash;
  }

  function renderPageContent() {
    if (activePage === "pipeline-overview") {
      return (
        <div className="space-y-6">
          <div className="space-y-4">
            <div>
              <h3 className="text-[22px] font-medium tracking-[-0.05em] text-[var(--foreground)]">Overview</h3>
            </div>
            <StatCards />
          </div>
          <div className="grid grid-cols-1 2xl:grid-cols-12 gap-6">
            <div className="2xl:col-span-8">
              <RobotViewer />
            </div>
            <div className="2xl:col-span-4 grid grid-cols-1 gap-6">
              <ProcessesWidget />
              <SensorsWidget />
            </div>
          </div>
          <PipelineFlow />
        </div>
      );
    }

    if (activePage === "planner-control") {
      return <NavigationControlPanel />;
    }

    if (activePage === "occupancy-map") {
      return <OccupancyMapPanel />;
    }

    if (activePage === "perception-memory") {
      return (
        <div className="grid grid-cols-1 2xl:grid-cols-2 gap-6">
          <PerceptionWidget />
          <MemoryWidget />
        </div>
      );
    }

    if (activePage === "ipc-viewer") {
      return (
        <div className="grid grid-cols-1 2xl:grid-cols-12 gap-6">
          <div className="2xl:col-span-8">
            <RobotViewer />
          </div>
          <div className="2xl:col-span-4 grid grid-cols-1 gap-6">
            <SensorsWidget />
            <IpcOrchestrationWidget />
          </div>
        </div>
      );
    }

    if (activePage === "external-services") {
      return (
        <div className="space-y-6">
          <ExternalServicesPanel />
          <ProcessesWidget />
        </div>
      );
    }

    if (activePage === "logs-events") {
      return <LogsWidget />;
    }

    if (activePage === "execution-modes") {
      return (
        <div className="space-y-6">
          <ControlStrip />
          <ExecutionModesPanel />
        </div>
      );
    }

    return <ArtifactsStoragePanel />;
  }

  return (
    <div className="dashboard-shell">
      {mobileSidebarOpen ? <button type="button" aria-label="Close navigation" className="dashboard-sidebar-backdrop" onClick={() => setMobileSidebarOpen(false)} /> : null}
      <Sidebar activePage={activePage} isMobileOpen={mobileSidebarOpen} onCloseMobile={() => setMobileSidebarOpen(false)} onNavigate={navigateTo} />
      <main className="dashboard-main">
        <TopBar page={page} onToggleSidebar={() => setMobileSidebarOpen((current) => !current)} />
        <div className="dashboard-page dashboard-scroll">
          <div className="dashboard-page-header">
            <div className="min-w-0">
              <div className="dashboard-eyebrow">{page.groupTitle}</div>
              <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-2">
                <h2 className="text-[28px] font-medium tracking-[-0.06em] text-[var(--foreground)]">{page.label}</h2>
                <span className="hidden h-4 w-px bg-[rgba(var(--ink-rgb),0.08)] lg:block" />
                <p className="dashboard-subtitle max-w-2xl">{page.description}</p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <ConsoleBadge tone={state?.session.active ? "emerald" : "amber"}>
                {state?.session.active ? "session live" : "session idle"}
              </ConsoleBadge>
              <ConsoleBadge tone="cyan" dot={false}>
                mode {String(state?.runtime.executionMode ?? state?.runtime.modes?.executionMode ?? "IDLE")}
              </ConsoleBadge>
              <ConsoleBadge tone="slate" dot={false}>
                scene {state?.session.config?.scenePreset ?? "draft"}
              </ConsoleBadge>
            </div>
          </div>

          <DashboardSectionTabs activePage={activePage} onNavigate={navigateTo} />

          <div className="dashboard-page-body">
            <div className="dashboard-page-primary">
              {error !== "" && (
                <div className="dashboard-panel border-[color:var(--tone-coral-border)] bg-[var(--tone-coral-bg)] px-4 py-3 text-[12px] text-[var(--tone-coral-fg)]">
                  {error}
                </div>
              )}

              {activePage === "pipeline-overview" ? null : <DashboardPageHero page={page} />}
              {renderPageContent()}
            </div>

            <DashboardActivityRail />
          </div>
        </div>
      </main>
    </div>
  );
}
