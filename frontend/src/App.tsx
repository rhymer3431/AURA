import { Bell, Database, Home, Network, Search } from 'lucide-react';
import { useState } from 'react';
import { LongTermMemory } from './components/LongTermMemory';
import { RealtimeStream } from './components/RealtimeStream';
import { SceneGraph } from './components/SceneGraph';
import { StreamingProvider } from './application/streaming/StreamingProvider';

type View = 'realtime' | 'memory' | 'graph';

export default function App() {
  const [activeView, setActiveView] = useState<View>('realtime');

  return (
    <StreamingProvider>
      <div className="relative flex h-screen w-full overflow-hidden bg-[#f8f9fa] font-['Helvetica',sans-serif]">

      {/* Left Sidebar */}
      <div className="fixed left-0 top-0 h-full w-[260px] bg-white shadow-[0px_3.5px_5.5px_0px_rgba(0,0,0,0.02)]">
        {/* Logo */}
        <div className="flex items-center gap-3 px-6 py-8">
          <div className="flex size-9 items-center justify-center rounded-xl bg-[#4fd1c5]">
            <Home className="size-5 text-white" />
          </div>
          <div className="text-[12px] text-[#2d3748] uppercase tracking-wide leading-tight">
            AURA PERCEPTION<br />DASHBOARD
          </div>
        </div>

        {/* Main Navigation */}
        <div className="mt-6 px-6 space-y-2">
          <button
            onClick={() => setActiveView('realtime')}
            className={`flex w-full items-center gap-3 rounded-xl px-4 py-3 transition-all ${activeView === 'realtime'
              ? 'bg-[#4fd1c5] text-white shadow-sm'
              : 'text-[#a0aec0] hover:bg-gray-50'
              }`}
          >
            <Home className="size-4" />
            <span className="text-[14px]">Realtime Stream</span>
          </button>

          <button
            onClick={() => setActiveView('memory')}
            className={`flex w-full items-center gap-3 rounded-xl px-4 py-3 transition-all ${activeView === 'memory'
              ? 'bg-[#4fd1c5] text-white shadow-sm'
              : 'text-[#a0aec0] hover:bg-gray-50'
              }`}
          >
            <Database className="size-4" />
            <span className="text-[14px]">Long-term Memory</span>
          </button>

          <button
            onClick={() => setActiveView('graph')}
            className={`flex w-full items-center gap-3 rounded-xl px-4 py-3 transition-all ${activeView === 'graph'
              ? 'bg-[#4fd1c5] text-white shadow-sm'
              : 'text-[#a0aec0] hover:bg-gray-50'
              }`}
          >
            <Network className="size-4" />
            <span className="text-[14px]">Scene Graph</span>
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="ml-[260px] flex h-full min-h-0 flex-col w-full ">

        {/* Top Bar */}
        <div className="shrink-0 bg-[#f8f9fa] pl-8 pr-10 py-6 flex items-center justify-between">

          <div className="ml-4 text-[18px] text-[#2d3748]">

            {activeView === 'realtime' && 'Realtime Stream'}
            {activeView === 'memory' && 'Long-term Memory'}
            {activeView === 'graph' && 'Scene Graph'}
          </div>

          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2 rounded-full border border-[#e2e8f0] bg-white px-4 py-2">
              <Search className="size-4 text-[#a0aec0]" />
              <input
                type="text"
                placeholder="Search entities or frames"
                className="w-56 border-0 bg-transparent text-[14px] text-[#2d3748] outline-none"
              />
            </div>
            <button className="text-[#a0aec0] transition-colors hover:text-[#2d3748]">
              <Bell className="size-5" />
            </button>
          </div>
        </div>

        {/* Dynamic Content Area */}
        <div className="
          flex-1 min-h-0 w-full
          px-6 md:px-10 lg:px-16 xl:px-24 2xl:px-32
          pb-2
        ">
          <div className="h-full overflow-auto pb-3">
            {activeView === 'realtime' && <RealtimeStream />}
            {activeView === 'memory' && <LongTermMemory />}
            {activeView === 'graph' && <SceneGraph />}
          </div>
        </div>

      </div>
      </div>
    </StreamingProvider>
  );
}
