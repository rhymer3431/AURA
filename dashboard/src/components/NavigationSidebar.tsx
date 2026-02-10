import { Database, Home, Map, Network, Settings2 } from 'lucide-react';
import type { ComponentType } from 'react';

export type View = 'pipeline' | 'realtime' | 'slam' | 'semantic' | 'habitat' | 'memory' | 'graph';

type NavigationSidebarProps = {
  activeView: View;
  onViewChange: (view: View) => void;
};

export function NavigationSidebar({
  activeView,
  onViewChange,
}: NavigationSidebarProps) {
  const renderButton = (view: View, label: string, Icon: ComponentType<{ className?: string }>) => (
    <button
      onClick={() => onViewChange(view)}
      className={`flex w-full items-center gap-3 rounded-xl px-4 py-3 transition-all ${activeView === view
        ? 'bg-[#4fd1c5] text-white shadow-sm'
        : 'text-[#a0aec0] hover:bg-gray-50'
        }`}
    >
      <Icon className="size-4" />
      <span className="text-[14px]">{label}</span>
    </button>
  );

  return (
    <div className="fixed left-0 top-0 z-20 h-full w-[260px] bg-white shadow-[0px_3.5px_5.5px_0px_rgba(0,0,0,0.02)]">
        <div className="flex items-center justify-between gap-3 px-6 py-8">
          <div className="flex items-center gap-3">
            <div className="flex size-9 items-center justify-center rounded-xl bg-[#4fd1c5]">
              <Home className="size-5 text-white" />
            </div>
            <div className="text-[16px] uppercase leading-tight tracking-wide text-[#2d3748]">
              AURA
            </div>
          </div>
        </div>

        <div className="mt-2 space-y-2 px-6">
          {renderButton('pipeline', 'Pipeline Manager', Settings2)}
          {renderButton('realtime', 'AURA Viewer', Home)}
          {renderButton('slam', 'SLAM Viewer', Network)}
          {renderButton('semantic', 'Semantic Map', Map)}
          {renderButton('habitat', 'Habitat Viewer', Home)}
          {renderButton('memory', 'Long-term Memory', Database)}
          {renderButton('graph', 'Scene Graph', Network)}
        </div>
      </div>
  );
}
