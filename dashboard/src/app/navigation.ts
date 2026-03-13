import type { LucideIcon } from "lucide-react";
import {
  Bot,
  Eye,
  FileText,
  LayoutDashboard,
  Navigation,
  Radio,
  Scan,
  Settings,
} from "lucide-react";

export type DashboardPageId =
  | "pipeline-overview"
  | "planner-control"
  | "perception-memory"
  | "ipc-viewer"
  | "external-services"
  | "logs-events"
  | "execution-modes"
  | "artifacts-storage";

export type DashboardPage = {
  id: DashboardPageId;
  label: string;
  description: string;
  groupTitle: string;
  icon: LucideIcon;
};

type DashboardNavSection = {
  title: string;
  items: DashboardPage[];
};

export const DEFAULT_DASHBOARD_PAGE: DashboardPageId = "pipeline-overview";

export const dashboardNavSections: DashboardNavSection[] = [
  {
    title: "Dashboards",
    items: [
      {
        id: "pipeline-overview",
        label: "Pipeline Overview",
        description: "핵심 런타임 상태와 전체 파이프라인을 한눈에 봅니다.",
        groupTitle: "Dashboards",
        icon: LayoutDashboard,
      },
      {
        id: "planner-control",
        label: "Planner & Control",
        description: "플래너 상태, trajectory freshness, action status를 집중해서 확인합니다.",
        groupTitle: "Dashboards",
        icon: Navigation,
      },
      {
        id: "perception-memory",
        label: "Perception & Memory",
        description: "탐지 파이프라인과 memory scratchpad 상태를 분리해서 봅니다.",
        groupTitle: "Dashboards",
        icon: Scan,
      },
    ],
  },
  {
    title: "Monitoring",
    items: [
      {
        id: "ipc-viewer",
        label: "IPC & Viewer",
        description: "WebRTC 뷰어, 센서 입력, IPC transport를 전용 페이지로 분리했습니다.",
        groupTitle: "Monitoring",
        icon: Eye,
      },
      {
        id: "external-services",
        label: "External Services",
        description: "NavDP, Dual, System2와 관련 프로세스 구성을 함께 모니터링합니다.",
        groupTitle: "Monitoring",
        icon: Radio,
      },
      {
        id: "logs-events",
        label: "Logs & Events",
        description: "시스템 로그와 이벤트 스트림을 별도 페이지에서 길게 확인합니다.",
        groupTitle: "Monitoring",
        icon: FileText,
      },
    ],
  },
  {
    title: "Configuration",
    items: [
      {
        id: "execution-modes",
        label: "Execution Modes",
        description: "세션 시작 옵션과 현재 실행 모드를 별도 설정 화면으로 분리했습니다.",
        groupTitle: "Configuration",
        icon: Bot,
      },
      {
        id: "artifacts-storage",
        label: "Artifacts & Storage",
        description: "로그 파일, endpoint, memory footprint 같은 운영 산출물을 모아 봅니다.",
        groupTitle: "Configuration",
        icon: Settings,
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
  return normalized in dashboardPages
    ? (normalized as DashboardPageId)
    : DEFAULT_DASHBOARD_PAGE;
}

export function dashboardPageHash(pageId: DashboardPageId): string {
  return `#/${pageId}`;
}
