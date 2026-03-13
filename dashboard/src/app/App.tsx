import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { StatCards } from "./components/StatCards";
import { PipelineFlow } from "./components/PipelineFlow";
import { NavigationControlPanel } from "./components/NavigationControlPanel";
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
import { useDashboard } from "./state";

export default function App() {
  const { error } = useDashboard();

  return (
    <div className="flex h-screen bg-white overflow-hidden font-['Inter',sans-serif]">
      {/* Left Sidebar */}
      <Sidebar />

      {/* Main Content */}
      <main className="flex-1 min-w-0 flex flex-col h-screen overflow-hidden">
        <TopBar />
        <div className="flex-1 overflow-y-auto px-8 pb-10">
          {/* Overview Header */}
          <div className="flex items-center justify-between mb-6 mt-6">
            <h2 className="text-[16px] font-semibold text-black">Overview</h2>
          </div>

          {error !== "" && (
            <div className="mb-4 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-[12px] text-red-700">
              {error}
            </div>
          )}

          <ControlStrip />

          {/* Stat Cards */}
          <div className="mt-6">
            <StatCards />
          </div>

          {/*
            Main Layout Grid: 2 columns structure
            Left Column: Flow & Core Panels (col-span-8 or 7)
            Right Column: Status Widgets (col-span-4 or 5)
          */}
          <div className="mt-6 grid grid-cols-1 xl:grid-cols-12 gap-6">
            {/* Left Column - Core Pipeline & Services (Spans 8 columns on large screens) */}
            <div className="xl:col-span-8 flex flex-col gap-6 min-w-0">
              {/* Robot View (Monitoring Viewer) */}
              <RobotViewer />

              {/* Pipeline Flow */}
              <PipelineFlow />

              {/* Navigation & Control */}
              <NavigationControlPanel />

              {/* External Services */}
              <ExternalServicesPanel />

              {/* Logs placed at the bottom of the left column for wide readability */}
              <LogsWidget />
            </div>

            {/* Right Column - Detailed System Metrics (Spans 4 columns) */}
            <div className="xl:col-span-4 flex flex-col gap-6 min-w-0">
              {/* System Health Section */}
              <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-1 gap-6">
                <ProcessesWidget />
                <SensorsWidget />
                <PerceptionWidget />
                <MemoryWidget />
                <IpcOrchestrationWidget />
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
