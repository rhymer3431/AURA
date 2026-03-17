import { useEffect, useState } from "react";

import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { OperationsPage } from "./components/OperationsPage";
import { NavigationPage } from "./components/NavigationPage";
import { DiagnosticsPage } from "./components/DiagnosticsPage";
import { SessionConfigPage } from "./components/SessionConfigPage";
import {
  PerceptionWidget,
  MemoryWidget,
} from "./components/SystemStatusWidgets";
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
    if (activePage === "operations") {
      return <OperationsPage />;
    }

    if (activePage === "perception-memory") {
      return (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <PerceptionWidget />
          <MemoryWidget />
        </div>
      );
    }

    if (activePage === "navigation") {
      return <NavigationPage />;
    }

    if (activePage === "diagnostics") {
      return <DiagnosticsPage />;
    }

    return <SessionConfigPage />;
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
