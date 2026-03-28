import type { LucideIcon } from "lucide-react";
import {
  Activity,
  FileText,
  Map,
  MonitorPlay,
  SlidersHorizontal,
} from "lucide-react";

export type DashboardPageId =
  | "live-loop"
  | "spatial-memory-map"
  | "runtime-health-recovery"
  | "logs-replay"
  | "session-config";

type LegacyDashboardPageId =
  | "pipeline-overview"
  | "planner-control"
  | "perception-memory"
  | "occupancy-map"
  | "ipc-viewer"
  | "external-services"
  | "logs-events"
  | "execution-modes"
  | "artifacts-storage";

const legacyPageRedirects: Record<LegacyDashboardPageId, DashboardPageId> = {
  "pipeline-overview": "live-loop",
  "planner-control": "live-loop",
  "perception-memory": "live-loop",
  "occupancy-map": "spatial-memory-map",
  "ipc-viewer": "live-loop",
  "external-services": "runtime-health-recovery",
  "logs-events": "logs-replay",
  "execution-modes": "session-config",
  "artifacts-storage": "runtime-health-recovery",
};

export type DashboardPage = {
  id: DashboardPageId;
  label: string;
  description: string;
  groupTitle: string;
  icon: LucideIcon;
};

export type DashboardNavSection = {
  title: string;
  items: DashboardPage[];
};

export const DEFAULT_DASHBOARD_PAGE: DashboardPageId = "live-loop";

export const dashboardNavSections: DashboardNavSection[] = [
  {
    title: "Operations",
    items: [
      {
        id: "live-loop",
        label: "Live Loop",
        description: "로봇 시점, 인지 루프, 판단, 이동 명령을 한 화면에서 폐루프로 관찰합니다.",
        groupTitle: "Operations",
        icon: MonitorPlay,
      },
      {
        id: "spatial-memory-map",
        label: "Spatial Memory & Map",
        description: "occupancy map 위에서 현재 pose, waypoint progress, 기억된 객체와 장소를 공간적으로 확인합니다.",
        groupTitle: "Operations",
        icon: Map,
      },
      {
        id: "runtime-health-recovery",
        label: "Runtime Health & Recovery",
        description: "process, service, transport, recovery state machine과 safe-stop 원인을 한 페이지에서 진단합니다.",
        groupTitle: "Operations",
        icon: Activity,
      },
    ],
  },
  {
    title: "Analysis",
    items: [
      {
        id: "logs-replay",
        label: "Logs & Replay",
        description: "event stream과 프레임 단위 cognition trace를 함께 검색하고 replay 관점으로 점검합니다.",
        groupTitle: "Analysis",
        icon: FileText,
      },
      {
        id: "session-config",
        label: "Session & Mode Config",
        description: "launch mode, scene preset, locomotion config, task control을 한 곳에서 관리합니다.",
        groupTitle: "Analysis",
        icon: SlidersHorizontal,
      },
    ],
  },
];

export const dashboardPages = Object.fromEntries(
  dashboardNavSections.flatMap((section) => section.items.map((item) => [item.id, item])),
) as Record<DashboardPageId, DashboardPage>;

export function parseDashboardPageId(value: string | null | undefined): DashboardPageId {
  const normalized = String(value ?? "")
    .trim()
    .replace(/^#/, "")
    .replace(/^\//, "")
    .replace(/\/+$/, "");

  if (normalized in dashboardPages) {
    return normalized as DashboardPageId;
  }
  if (normalized in legacyPageRedirects) {
    return legacyPageRedirects[normalized as LegacyDashboardPageId];
  }
  return DEFAULT_DASHBOARD_PAGE;
}

export function dashboardPageHash(pageId: DashboardPageId): string {
  return `#/${pageId}`;
}
