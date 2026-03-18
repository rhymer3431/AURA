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
  MainControlServerWidget,
  ProcessesWidget,
  SensorsWidget,
  PerceptionWidget,
  MemoryWidget,
  IpcOrchestrationWidget,
  LogsWidget,
} from "./components/SystemStatusWidgets";
import { ExecutionModesPanel } from "./components/ExecutionModesPanel";
import { ArtifactsStoragePanel } from "./components/ArtifactsStoragePanel";
import {
  DEFAULT_DASHBOARD_PAGE,
  dashboardPageHash,
  dashboardPages,
  parseDashboardPageId,
  type DashboardPageId,
} from "./navigation";
import { useDashboard } from "./state";

function currentPageFromLocation(): DashboardPageId {
  if (typeof window === "undefined") {
    return DEFAULT_DASHBOARD_PAGE;
  }
  return parseDashboardPageId(window.location.hash);
}

export default function App() {
  const { error } = useDashboard();
  const [activePage, setActivePage] = useState<DashboardPageId>(() => currentPageFromLocation());

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

  const page = dashboardPages[activePage];

  function navigateTo(pageId: DashboardPageId) {
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
          <StatCards />
          <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
            <div className="xl:col-span-8">
              <RobotViewer />
            </div>
            <div className="xl:col-span-4 grid grid-cols-1 gap-6">
              <SensorsWidget />
              <MainControlServerWidget />
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
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <PerceptionWidget />
          <MemoryWidget />
        </div>
      );
    }

    if (activePage === "ipc-viewer") {
      return (
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
          <div className="xl:col-span-8">
            <RobotViewer />
          </div>
          <div className="xl:col-span-4 grid grid-cols-1 gap-6">
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
    <div className="flex h-screen bg-white overflow-hidden font-['Inter',sans-serif]">
      <Sidebar activePage={activePage} onNavigate={navigateTo} />
      <main className="flex-1 min-w-0 flex flex-col h-screen overflow-hidden">
        <TopBar page={page} />
        <div className="flex-1 overflow-y-auto px-8 pb-10">
          <div className="flex items-center justify-between mb-6 mt-6">
            <div>
              <h2 className="text-[18px] font-semibold text-black">{page.label}</h2>
              <p className="text-[12px] text-black/50 mt-1">{page.description}</p>
            </div>
          </div>

          {error !== "" && (
            <div className="mb-4 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-[12px] text-red-700">
              {error}
            </div>
          )}

          {renderPageContent()}
        </div>
      </main>
    </div>
  );
}
