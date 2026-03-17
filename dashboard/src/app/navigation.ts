import type { LucideIcon } from "lucide-react";
import {
  Activity,
  Navigation,
  Scan,
  ShieldAlert,
  Settings,
} from "lucide-react";

export type DashboardPageId =
  | "operations"
  | "navigation"
  | "perception-memory"
  | "diagnostics"
  | "session-config";

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

export const DEFAULT_DASHBOARD_PAGE: DashboardPageId = "operations";

export const dashboardNavSections: DashboardNavSection[] = [
  {
    title: "Operate",
    items: [
      {
        id: "operations",
        label: "Operations",
        description: "실시간 운영 판단에 필요한 핵심 상태와 라이브 상황을 한 화면에서 봅니다.",
        groupTitle: "Operate",
        icon: Activity,
      },
      {
        id: "navigation",
        label: "Navigation",
        description: "경로 진행, 목표 거리, 맵 기반 위치와 플래너 상태를 함께 확인합니다.",
        groupTitle: "Operate",
        icon: Navigation,
      },
    ],
  },
  {
    title: "Investigate",
    items: [
      {
        id: "perception-memory",
        label: "Perception & Memory",
        description: "인지 품질과 memory task context를 같은 맥락에서 분석합니다.",
        groupTitle: "Investigate",
        icon: Scan,
      },
      {
        id: "diagnostics",
        label: "Diagnostics",
        description: "서비스, IPC, 로그, route debug를 분리된 진단 화면에서 확인합니다.",
        groupTitle: "Investigate",
        icon: ShieldAlert,
      },
    ],
  },
  {
    title: "Configure",
    items: [
      {
        id: "session-config",
        label: "Session & Config",
        description: "세션 시작 옵션, 적용 중 설정, 제어 액션을 한곳에서 관리합니다.",
        groupTitle: "Configure",
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
