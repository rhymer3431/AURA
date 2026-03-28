import { useEffect, useState } from "react";

import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { LiveLoopWorkspace } from "./components/LiveLoopWorkspace";
import { SpatialMemoryMapWorkspace } from "./components/SpatialMemoryMapWorkspace";
import { RuntimeHealthRecoveryWorkspace } from "./components/RuntimeHealthRecoveryWorkspace";
import { LogsReplayWorkspace } from "./components/LogsReplayWorkspace";
import { SessionConfigWorkspace } from "./components/SessionConfigWorkspace";
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
    if (activePage === "live-loop") {
      return <LiveLoopWorkspace />;
    }
    if (activePage === "spatial-memory-map") {
      return <SpatialMemoryMapWorkspace />;
    }
    if (activePage === "runtime-health-recovery") {
      return <RuntimeHealthRecoveryWorkspace />;
    }
    if (activePage === "logs-replay") {
      return <LogsReplayWorkspace />;
    }
    return <SessionConfigWorkspace />;
  }

  return (
    <div className="dashboard-shell dashboard-shell--no-rail">
      {mobileSidebarOpen ? (
        <button
          type="button"
          aria-label="Close navigation"
          className="dashboard-sidebar-backdrop"
          onClick={() => setMobileSidebarOpen(false)}
        />
      ) : null}
      <Sidebar
        activePage={activePage}
        isMobileOpen={mobileSidebarOpen}
        onCloseMobile={() => setMobileSidebarOpen(false)}
        onNavigate={navigateTo}
      />
      <main className="dashboard-main">
        <TopBar page={page} onToggleSidebar={() => setMobileSidebarOpen((current) => !current)} />
        <div className="dashboard-page dashboard-scroll">
          <div className="dashboard-page-header">
            <div className="min-w-0">
              <div className="dashboard-eyebrow">{page.groupTitle}</div>
              <h1 className="dashboard-page-title mt-1">{page.label}</h1>
              <p className="dashboard-page-caption mt-1.5 max-w-xl">{page.description}</p>
            </div>
          </div>

          <div className="dashboard-page-body">
            {error !== "" && (
              <div className="dashboard-panel border-[color:var(--tone-coral-border)] bg-[var(--tone-coral-bg)] px-3.5 py-2.5 text-[12px] text-[var(--tone-coral-fg)]">
                {error}
              </div>
            )}

            {renderPageContent()}
          </div>
        </div>
      </main>
    </div>
  );
}
