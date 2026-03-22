import { useEffect, useState } from "react";
import { motion } from "motion/react";

import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
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
import { OverviewCanvas } from "./components/OverviewCanvas";
import {
  DEFAULT_DASHBOARD_PAGE,
  dashboardPageHash,
  dashboardPages,
  parseDashboardPageId,
  type DashboardPageId,
} from "./navigation";
import { asRecord } from "./selectors";
import { useDashboard } from "./state";
import { ConsoleBadge } from "./components/console-ui";
import { RightRail } from "./components/RightRail";

function currentPageFromLocation(): DashboardPageId {
  if (typeof window === "undefined") {
    return DEFAULT_DASHBOARD_PAGE;
  }
  return parseDashboardPageId(window.location.hash);
}

export default function App() {
  const { error, refresh, state } = useDashboard();
  const [activePage, setActivePage] = useState<DashboardPageId>(() => currentPageFromLocation());
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const runtime = asRecord(state?.runtime);
  const runtimeModes = asRecord(runtime.modes);

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
      return <OverviewCanvas />;
    }

    if (activePage === "planner-control") {
      return <NavigationControlPanel />;
    }

    if (activePage === "occupancy-map") {
      return <OccupancyMapPanel />;
    }

    if (activePage === "perception-memory") {
      return (
        <div className="grid grid-cols-1 gap-4 2xl:grid-cols-2">
          <PerceptionWidget />
          <MemoryWidget />
        </div>
      );
    }

    if (activePage === "ipc-viewer") {
      return (
        <div className="grid grid-cols-1 gap-4 2xl:grid-cols-12">
          <div className="2xl:col-span-8">
            <RobotViewer />
          </div>
          <div className="grid grid-cols-1 gap-4 2xl:col-span-4">
            <SensorsWidget />
            <IpcOrchestrationWidget />
          </div>
        </div>
      );
    }

    if (activePage === "external-services") {
      return (
        <div className="space-y-4">
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
        <div className="space-y-4">
          <ControlStrip />
          <ExecutionModesPanel />
        </div>
      );
    }

    return <ArtifactsStoragePanel />;
  }

  return (
    <div className="dashboard-shell">
      {mobileSidebarOpen ? (
        <button
          type="button"
          aria-label="Close navigation"
          className="dashboard-sidebar-backdrop"
          onClick={() => setMobileSidebarOpen(false)}
        />
      ) : null}
      <Sidebar activePage={activePage} isMobileOpen={mobileSidebarOpen} onCloseMobile={() => setMobileSidebarOpen(false)} onNavigate={navigateTo} />
      <main className="dashboard-main">
        <TopBar
          page={page}
          onToggleSidebar={() => setMobileSidebarOpen((current) => !current)}
          onRefresh={() => void refresh()}
        />
        <div className="dashboard-page dashboard-scroll">
          <div className="dashboard-page-header">
            <motion.div
              className="min-w-0"
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
            >
              <div className="dashboard-eyebrow">{page.groupTitle}</div>
              <h1 className="dashboard-page-title mt-1">{page.label}</h1>
              <p className="dashboard-page-caption mt-1.5 max-w-xl">{page.description}</p>
            </motion.div>

            <motion.div
              className="flex flex-wrap items-center gap-1.5"
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.06, ease: [0.22, 1, 0.36, 1] }}
            >
              <ConsoleBadge tone={state?.session.active ? "emerald" : "amber"}>
                {state?.session.active ? "session live" : "session idle"}
              </ConsoleBadge>
              <ConsoleBadge tone="cyan" dot={false}>
                mode {String(runtime.executionMode ?? runtimeModes.executionMode ?? "IDLE")}
              </ConsoleBadge>
              {state?.session.config?.scenePreset ? (
                <ConsoleBadge tone="violet" dot={false}>
                  scene {state.session.config.scenePreset}
                </ConsoleBadge>
              ) : null}
            </motion.div>
          </div>

          <div className="dashboard-page-body">
            {error !== "" && (
              <div className="dashboard-panel border-[color:var(--tone-coral-border)] bg-[var(--tone-coral-bg)] px-3.5 py-2.5 text-[12px] text-[var(--tone-coral-fg)]">
                {error}
              </div>
            )}

            {renderPageContent()}
            <RightRail mobile className="xl:hidden" />
          </div>
        </div>
      </main>
      <RightRail className="hidden xl:flex" />
    </div>
  );
}
