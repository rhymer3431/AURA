import {
  LayoutDashboard,
  Navigation,
  Scan,
  Eye,
  Radio,
  FileText,
  Bot,
  Settings,
} from "lucide-react";

const navSections = [
  {
    title: "Dashboards",
    items: [
      { icon: LayoutDashboard, label: "Pipeline Overview", active: true },
      { icon: Navigation, label: "Planner & Control", active: false },
      { icon: Scan, label: "Perception & Memory", active: false },
    ],
  },
  {
    title: "Monitoring",
    items: [
      { icon: Eye, label: "IPC & Viewer", active: false },
      { icon: Radio, label: "External Services", active: false },
      { icon: FileText, label: "Logs & Events", active: false },
    ],
  },
  {
    title: "Configuration",
    items: [
      { icon: Bot, label: "Execution Modes", active: false },
      { icon: Settings, label: "Artifacts & Storage", active: false },
    ],
  },
];

export function Sidebar() {
  return (
    <aside className="w-[240px] min-w-[240px] border-r border-black/5 flex flex-col h-screen bg-white font-['Inter',sans-serif]">
      <div className="flex items-center gap-2.5 px-6 pt-6 pb-6">
        <div className="size-8 rounded-full shrink-0 bg-black text-white flex items-center justify-center">
          <Bot className="size-4" />
        </div>
        <span className="text-[14px] font-medium text-black">AURA Sys</span>
      </div>

      <nav className="flex-1 overflow-y-auto px-4 pb-6 space-y-6">
        {navSections.map((section) => (
          <div key={section.title}>
            <div className="px-2 mb-2">
              <span className="text-[13px] text-black/40">{section.title}</span>
            </div>
            <div className="space-y-0.5">
              {section.items.map((item) => (
                <div
                  key={item.label}
                  className={`flex items-center gap-3 px-2 py-1.5 rounded-xl cursor-pointer transition-colors ${
                    item.active
                      ? "bg-black/[0.04] text-black"
                      : "text-black hover:bg-black/[0.02]"
                  }`}
                >
                  <div className={`flex items-center justify-center ${item.active ? "text-black" : "text-black/40"}`}>
                    <item.icon className="size-5" strokeWidth={item.active ? 2 : 1.5} />
                  </div>
                  <span className={`text-[14px] ${item.active ? "font-medium" : ""}`}>{item.label}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </nav>

      <div className="p-6 border-t border-black/5 mt-auto">
        <div className="flex items-center gap-2 text-black/40 justify-center">
          <Bot className="size-4" />
          <span className="text-[12px] font-semibold tracking-wider">AURA UI</span>
        </div>
      </div>
    </aside>
  );
}
