import { Power } from 'lucide-react';
import { useState } from 'react';
import { StreamingProvider } from './application/streaming/StreamingProvider';
import { STREAM_BASE_URL } from './application/streaming/streamConfig';
import { HabitatViewer } from './components/HabitatViewer';
import { LongTermMemory } from './components/LongTermMemory';
import { NavigationSidebar, type View } from './components/NavigationSidebar';
import { PipelineManager } from './components/PipelineManager';
import { RealtimeStream } from './components/RealtimeStream';
import { SceneGraph } from './components/SceneGraph';
import { SemanticMapViewer } from './components/SemanticMapViewer';
import { SlamViewer } from './components/SlamViewer';

export default function App() {
  const [activeView, setActiveView] = useState<View>('pipeline');
  const [isShuttingDown, setIsShuttingDown] = useState(false);
  const [shutdownNotice, setShutdownNotice] = useState<string | null>(null);

  const shutdownEndpoint = `${STREAM_BASE_URL.replace(/\/$/, '')}/system/server/shutdown`;

  const handleViewChange = (view: View) => {
    setActiveView(view);
  };

  const handleShutdown = async () => {
    if (isShuttingDown) return;
    const confirmed = window.confirm('백엔드 서버를 종료하고 대시보드를 닫을까요?');
    if (!confirmed) return;

    setIsShuttingDown(true);
    setShutdownNotice(null);

    try {
      const response = await fetch(shutdownEndpoint, { method: 'POST' });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      setShutdownNotice('백엔드 종료를 요청했습니다. 대시보드를 닫습니다.');

      window.setTimeout(() => {
        window.close();
      }, 1000);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setShutdownNotice(`종료 요청 실패: ${message}`);
      setIsShuttingDown(false);
    }
  };

  return (
    <StreamingProvider>
      <div className="relative flex h-dvh w-full overflow-hidden bg-[#f8f9fa] font-['Helvetica',sans-serif]">

        <NavigationSidebar
          activeView={activeView}
          onViewChange={handleViewChange}
        />

        {/* Main Content Area */}
        <div className="ml-[260px] flex h-full min-h-0 w-full flex-col">

          {/* Top Bar - Compact */}
          <div className="flex shrink-0 items-center border-b border-[#e2e8f0] bg-[#f8f9fa] px-6 py-3">
            <div className="flex w-full items-center justify-between gap-2">
              <div className="flex min-w-0 items-center">
                <div className="truncate text-[16px] font-medium text-[#2d3748]">
                  {activeView === 'pipeline' && 'Pipeline Manager'}
                  {activeView === 'realtime' && 'AURA Viewer'}
                  {activeView === 'slam' && 'SLAM Viewer'}
                  {activeView === 'semantic' && 'Semantic Map Viewer'}
                  {activeView === 'habitat' && 'Habitat Viewer'}
                  {activeView === 'memory' && 'Long-term Memory'}
                  {activeView === 'graph' && 'Scene Graph'}
                </div>
              </div>

              <div className="flex items-center gap-3">
                {shutdownNotice && (
                  <span className="text-xs text-[#4a5568]">
                    {shutdownNotice}
                  </span>
                )}
                <button
                  className="inline-flex items-center gap-2 rounded-md border border-[#fed7d7] bg-[#fff5f5] px-3 py-2 text-sm font-medium text-[#c53030] transition-colors hover:bg-[#fff0f0] disabled:cursor-not-allowed disabled:opacity-70"
                  onClick={handleShutdown}
                  disabled={isShuttingDown}
                  aria-label="Shutdown backend server and close dashboard"
                >
                  <Power className="size-4" />
                  {isShuttingDown ? '종료 중...' : '종료'}
                </button>
              </div>
            </div>
          </div>

          {/* Dynamic Content Area */}
          <div className="flex-1 min-h-0 w-full px-10 pb-2">
            <div className="h-full overflow-auto pb-3">
              {activeView === 'pipeline' && <PipelineManager />}
              {activeView === 'realtime' && <RealtimeStream />}
              {activeView === 'slam' && <SlamViewer />}
              {activeView === 'semantic' && <SemanticMapViewer />}
              {activeView === 'habitat' && <HabitatViewer />}
              {activeView === 'memory' && <LongTermMemory />}
              {activeView === 'graph' && <SceneGraph />}
            </div>
          </div>

        </div>
      </div>
    </StreamingProvider>
  );
}
