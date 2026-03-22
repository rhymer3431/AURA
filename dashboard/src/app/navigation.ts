import type { LucideIcon } from "lucide-react";
import {
  Bot,
  Eye,
  FileText,
  LayoutDashboard,
  Map,
  Navigation,
  Radio,
  Scan,
  Settings,
} from "lucide-react";

export type DashboardPageId =
  | "pipeline-overview"
  | "planner-control"
  | "perception-memory"
  | "occupancy-map"
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

export type DashboardNavSection = {
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
        description: "운영 파이프라인 핵심 신호와 라이브 비전, 프로세스 구성을 한 화면에서 확인합니다.",
        groupTitle: "Dashboards",
        icon: LayoutDashboard,
      },
      {
        id: "planner-control",
        label: "Planner & Control",
        description: "planning context, recovery state, command arbitration을 집중해서 확인합니다.",
        groupTitle: "Dashboards",
        icon: Navigation,
      },
      {
        id: "perception-memory",
        label: "Perception & Memory",
        description: "snapshot-backed Perception / Memory 모듈 상태를 분리해서 봅니다.",
        groupTitle: "Dashboards",
        icon: Scan,
      },
      {
        id: "occupancy-map",
        label: "Occupancy Map",
        description: "맵 기반 occupancy 뷰에서 현재 로봇 위치와 전역 경로를 실시간으로 겹쳐 봅니다.",
        groupTitle: "Dashboards",
        icon: Map,
      },
    ],
  },
  {
    title: "Monitoring",
    items: [
      {
        id: "ipc-viewer",
        label: "IPC & Viewer",
        description: "WebRTC 뷰어, gateway ingress, telemetry mirror 상태를 전용 페이지에서 확인합니다.",
        groupTitle: "Monitoring",
        icon: Eye,
      },
      {
        id: "external-services",
        label: "External Services",
        description: "외부 서비스와 주요 모듈의 health, latency mirror를 한 번에 봅니다.",
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
        description: "runtime entry mode와 현재 세션 설정을 별도 화면에서 비교합니다.",
        groupTitle: "Configuration",
        icon: Bot,
      },
      {
        id: "artifacts-storage",
        label: "Artifacts & Storage",
        description: "runtime artifacts, endpoints, raw process logs 같은 구현 진단 정보를 모아 봅니다.",
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
