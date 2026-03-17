import { useEffect, useMemo, useState } from "react";
import { Crosshair, Map, Route, Target } from "lucide-react";

import { useDashboard } from "../state";
import { buildApiUrl, requestJson } from "../network";
import { asArray, asRecord, numberValue, stringValue } from "../selectors";

type OccupancyMapResponse = {
  available: boolean;
  scenePreset: string;
  canonicalScenePreset?: string;
  label?: string;
  imagePath?: string;
  imageUrl?: string;
  imageWidth?: number;
  imageHeight?: number;
  xMin?: number;
  xMax?: number;
  yMin?: number;
  yMax?: number;
  resolutionMpp?: number;
  reason?: string;
};

type CanvasPoint = {
  x: number;
  y: number;
  inBounds: boolean;
};

function asVec2(value: unknown): [number, number] | null {
  const items = asArray(value);
  if (items.length < 2) {
    return null;
  }
  const x = numberValue(items[0]);
  const y = numberValue(items[1]);
  return x === null || y === null ? null : [x, y];
}

function asWorldPolyline(value: unknown): [number, number][] {
  return asArray(value)
    .map((entry) => asVec2(entry))
    .filter((entry): entry is [number, number] => entry !== null);
}

function mapWorldToCanvas(meta: OccupancyMapResponse | null, point: [number, number] | null): CanvasPoint | null {
  if (
    meta === null ||
    meta.available !== true ||
    point === null ||
    typeof meta.xMin !== "number" ||
    typeof meta.xMax !== "number" ||
    typeof meta.yMin !== "number" ||
    typeof meta.yMax !== "number" ||
    typeof meta.resolutionMpp !== "number"
  ) {
    return null;
  }
  const width = Math.max(Number(meta.imageWidth ?? 0) - 1, 1);
  const height = Math.max(Number(meta.imageHeight ?? 0) - 1, 1);
  const col = (point[0] - meta.xMin) / meta.resolutionMpp;
  const row = (meta.yMax - point[1]) / meta.resolutionMpp;
  const clampedX = Math.max(0, Math.min(width, col));
  const clampedY = Math.max(0, Math.min(height, row));
  const inBounds = point[0] >= meta.xMin && point[0] <= meta.xMax && point[1] >= meta.yMin && point[1] <= meta.yMax;
  return { x: clampedX, y: clampedY, inBounds };
}

function formatCoord(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(2) : "n/a";
}

type OccupancyMapPanelProps = {
  showRouteDebug?: boolean;
};

export function OccupancyMapPanel({ showRouteDebug = true }: OccupancyMapPanelProps) {
  const { state, form } = useDashboard();
  const [mapData, setMapData] = useState<OccupancyMapResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState("");

  const activeScenePreset = state?.session.config?.scenePreset ?? form.scenePreset;
  const runtime = asRecord(state?.runtime);
  const sensors = asRecord(state?.sensors);

  useEffect(() => {
    let cancelled = false;
    const scenePreset = String(activeScenePreset ?? "").trim();
    if (scenePreset === "") {
      setMapData(null);
      setLoadError("");
      return undefined;
    }
    setLoading(true);
    setLoadError("");
    requestJson<OccupancyMapResponse>(`/api/occupancy/current?scenePreset=${encodeURIComponent(scenePreset)}`)
      .then((payload) => {
        if (!cancelled) {
          setMapData(payload);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setMapData(null);
          setLoadError(error instanceof Error ? error.message : String(error));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [activeScenePreset]);

  const robotPose = asArray<number>(sensors.robotPoseXyz);
  const robotYawRad = numberValue(sensors.robotYawRad) ?? 0;
  const routeGoal = asVec2(runtime.globalRouteGoalXy) ?? (
    state?.session.config?.goal ? [state.session.config.goal.x, state.session.config.goal.y] as [number, number] : null
  );
  const activeWaypoint = asVec2(runtime.globalRouteActiveWaypointXy);
  const polyline = asWorldPolyline(runtime.globalRouteWaypointsWorld);

  const robotCanvas = mapWorldToCanvas(mapData, robotPose.length >= 2 ? [robotPose[0], robotPose[1]] : null);
  const goalCanvas = mapWorldToCanvas(mapData, routeGoal);
  const activeWaypointCanvas = mapWorldToCanvas(mapData, activeWaypoint);
  const routeCanvas = useMemo(() => polyline.map((point) => mapWorldToCanvas(mapData, point)).filter((point): point is CanvasPoint => point !== null), [mapData, polyline]);

  const svgRoutePoints = routeCanvas.map((point) => `${point.x},${point.y}`).join(" ");
  const imageSrc =
    typeof mapData?.imagePath === "string" && mapData.imagePath !== ""
      ? buildApiUrl(mapData.imagePath)
      : typeof mapData?.imageUrl === "string"
        ? mapData.imageUrl
        : "";
  const headingDeg = (-robotYawRad * 180) / Math.PI;
  const routeEnabled = Boolean(runtime.globalRouteEnabled);
  const routeStatus = routeEnabled ? (Boolean(runtime.globalRouteActive) ? "active" : "idle") : "disabled";

  return (
    <div className="bg-[#F5F8FB] rounded-[28px] p-6">
      <div className="flex items-center justify-between gap-4 mb-5">
        <div>
          <div className="flex items-center gap-2 text-[12px] text-black/45 mb-1">
            <Map className="size-4" />
            Occupancy Localization
          </div>
          <h3 className="text-[18px] font-semibold text-black">Occupancy Map</h3>
          <p className="text-[12px] text-black/45 mt-1">
            scene: <span className="text-black/70">{activeScenePreset || "inactive"}</span>
          </p>
        </div>
        <div className="grid grid-cols-2 gap-3 min-w-[280px]">
          <div className="rounded-2xl bg-white border border-black/5 px-4 py-3">
            <div className="text-[11px] text-black/45 mb-1">Robot XY</div>
            <div className="text-[15px] font-medium text-black">
              {formatCoord(robotPose[0])}, {formatCoord(robotPose[1])}
            </div>
          </div>
          <div className="rounded-2xl bg-white border border-black/5 px-4 py-3">
            <div className="text-[11px] text-black/45 mb-1">Global Route</div>
            <div className="text-[15px] font-medium text-black">
              {routeStatus} {routeEnabled ? `(${Number(runtime.globalRouteWaypointIndex ?? 0)}/${Number(runtime.globalRouteWaypointCount ?? 0)})` : ""}
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        <div className="xl:col-span-8">
          <div className="rounded-[24px] overflow-hidden border border-black/8 bg-white">
            <div className="px-4 py-3 border-b border-black/6 flex items-center justify-between bg-[linear-gradient(135deg,#fff8ee_0%,#f9fbff_100%)]">
              <div className="flex items-center gap-2 text-[12px] text-black/55">
                <Route className="size-4" />
                {mapData?.label ?? "Occupancy source"}
              </div>
              <div className="text-[11px] text-black/40">
                res {typeof mapData?.resolutionMpp === "number" ? `${mapData.resolutionMpp.toFixed(3)} m/px` : "n/a"}
              </div>
            </div>

            {loading && <div className="px-5 py-10 text-[13px] text-black/50">occupancy map metadata를 불러오는 중입니다.</div>}

            {!loading && (loadError !== "" || mapData?.available === false || mapData === null) && (
              <div className="px-5 py-10 text-[13px] text-black/55">
                <div className="font-medium text-black mb-2">이 scene에는 occupancy map 표시를 준비하지 못했습니다.</div>
                <div>{loadError || mapData?.reason || "선택된 scene preset에 대응하는 occupancy 자산이 없습니다."}</div>
              </div>
            )}

            {!loading && mapData?.available === true && (
              <div className="p-4">
                <div
                  className="relative w-full overflow-hidden rounded-[20px] border border-black/10 bg-[#eef3f8]"
                  style={{ aspectRatio: `${mapData.imageWidth ?? 1} / ${mapData.imageHeight ?? 1}` }}
                >
                  <img
                    src={imageSrc}
                    alt={`Occupancy map for ${mapData.label ?? activeScenePreset}`}
                    className="absolute inset-0 h-full w-full object-contain"
                  />
                  <svg
                    className="absolute inset-0 h-full w-full"
                    viewBox={`0 0 ${mapData.imageWidth ?? 1} ${mapData.imageHeight ?? 1}`}
                    preserveAspectRatio="none"
                  >
                    <defs>
                      <filter id="routeGlow">
                        <feGaussianBlur stdDeviation="1.8" result="blur" />
                        <feMerge>
                          <feMergeNode in="blur" />
                          <feMergeNode in="SourceGraphic" />
                        </feMerge>
                      </filter>
                    </defs>
                    {svgRoutePoints !== "" && (
                      <polyline
                        points={svgRoutePoints}
                        fill="none"
                        stroke="rgba(241, 103, 24, 0.92)"
                        strokeWidth={3}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        filter="url(#routeGlow)"
                      />
                    )}
                    {goalCanvas !== null && (
                      <g transform={`translate(${goalCanvas.x},${goalCanvas.y})`}>
                        <rect x={-7} y={-7} width={14} height={14} transform="rotate(45)" fill="#0f172a" stroke="#ffffff" strokeWidth={2} />
                      </g>
                    )}
                    {activeWaypointCanvas !== null && (
                      <g transform={`translate(${activeWaypointCanvas.x},${activeWaypointCanvas.y})`}>
                        <circle r={8} fill="rgba(14, 165, 233, 0.20)" />
                        <circle r={4} fill="#0284c7" stroke="#ffffff" strokeWidth={1.5} />
                      </g>
                    )}
                    {robotCanvas !== null && (
                      <g transform={`translate(${robotCanvas.x},${robotCanvas.y}) rotate(${headingDeg})`}>
                        <circle r={12} fill="rgba(16, 185, 129, 0.16)" />
                        <path d="M 11 0 L -7 6 L -3 0 L -7 -6 Z" fill="#059669" stroke="#ffffff" strokeWidth={1.5} />
                      </g>
                    )}
                  </svg>
                </div>

                <div className="mt-4 flex flex-wrap items-center gap-3 text-[11px] text-black/50">
                  <div className="inline-flex items-center gap-2 rounded-full bg-[#fff6ee] px-3 py-1.5">
                    <span className="size-2 rounded-full bg-[#f16718]" />
                    route
                  </div>
                  <div className="inline-flex items-center gap-2 rounded-full bg-[#eef8ff] px-3 py-1.5">
                    <span className="size-2 rounded-full bg-[#0284c7]" />
                    active waypoint
                  </div>
                  <div className="inline-flex items-center gap-2 rounded-full bg-[#eefbf6] px-3 py-1.5">
                    <span className="size-2 rounded-full bg-[#059669]" />
                    robot
                  </div>
                  <div className="inline-flex items-center gap-2 rounded-full bg-[#f3f4f6] px-3 py-1.5">
                    <span className="size-2 rotate-45 bg-[#0f172a]" />
                    final goal
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="xl:col-span-4 space-y-4">
          <div className="rounded-[24px] bg-white border border-black/6 p-4">
            <div className="flex items-center gap-2 text-[12px] text-black/45 mb-3">
              <Crosshair className="size-4" />
              Localization Snapshot
            </div>
            <div className="grid grid-cols-2 gap-3 text-[12px]">
              <div className="rounded-2xl bg-[#F7F9FB] px-3 py-3">
                <div className="text-black/45 mb-1">Robot X</div>
                <div className="text-black font-medium">{formatCoord(robotPose[0])}</div>
              </div>
              <div className="rounded-2xl bg-[#F7F9FB] px-3 py-3">
                <div className="text-black/45 mb-1">Robot Y</div>
                <div className="text-black font-medium">{formatCoord(robotPose[1])}</div>
              </div>
              <div className="rounded-2xl bg-[#F7F9FB] px-3 py-3">
                <div className="text-black/45 mb-1">Yaw</div>
                <div className="text-black font-medium">{robotYawRad.toFixed(2)} rad</div>
              </div>
              <div className="rounded-2xl bg-[#F7F9FB] px-3 py-3">
                <div className="text-black/45 mb-1">Goal Dist</div>
                <div className="text-black font-medium">
                  {typeof runtime.goalDistanceM === "number" ? `${Number(runtime.goalDistanceM).toFixed(2)}m` : "n/a"}
                </div>
              </div>
            </div>
          </div>

          {showRouteDebug ? (
            <div className="rounded-[24px] bg-white border border-black/6 p-4">
              <div className="flex items-center gap-2 text-[12px] text-black/45 mb-3">
                <Target className="size-4" />
                Route Debug
              </div>
              <div className="space-y-3 text-[12px]">
                <div className="rounded-2xl bg-[#F7F9FB] px-3 py-3">
                  <div className="text-black/45 mb-1">Replan reason</div>
                  <div className="text-black font-medium">{stringValue(runtime.globalRouteLastReplanReason, "none") || "none"}</div>
                </div>
                <div className="rounded-2xl bg-[#F7F9FB] px-3 py-3">
                  <div className="text-black/45 mb-1">Route error</div>
                  <div className="text-black font-medium break-words">{stringValue(runtime.globalRouteLastError, "clear") || "clear"}</div>
                </div>
                <div className="rounded-2xl bg-[#F7F9FB] px-3 py-3">
                  <div className="text-black/45 mb-1">Map bounds</div>
                  <div className="text-black font-medium">
                    x [{formatCoord(mapData?.xMin)}, {formatCoord(mapData?.xMax)}]
                  </div>
                  <div className="text-black font-medium">
                    y [{formatCoord(mapData?.yMin)}, {formatCoord(mapData?.yMax)}]
                  </div>
                </div>
                <div className="rounded-2xl bg-[#F7F9FB] px-3 py-3">
                  <div className="text-black/45 mb-1">Active waypoint</div>
                  <div className="text-black font-medium">
                    {activeWaypoint ? `${activeWaypoint[0].toFixed(2)}, ${activeWaypoint[1].toFixed(2)}` : "none"}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="rounded-[24px] bg-white border border-black/6 p-4">
              <div className="flex items-center gap-2 text-[12px] text-black/45 mb-3">
                <Target className="size-4" />
                Route Progress
              </div>
              <div className="space-y-3 text-[12px]">
                <div className="rounded-2xl bg-[#F7F9FB] px-3 py-3">
                  <div className="text-black/45 mb-1">Route status</div>
                  <div className="text-black font-medium">{routeStatus}</div>
                </div>
                <div className="rounded-2xl bg-[#F7F9FB] px-3 py-3">
                  <div className="text-black/45 mb-1">Waypoint progress</div>
                  <div className="text-black font-medium">
                    {Number(runtime.globalRouteWaypointIndex ?? 0)} / {Number(runtime.globalRouteWaypointCount ?? 0)}
                  </div>
                </div>
                <div className="rounded-2xl bg-[#F7F9FB] px-3 py-3">
                  <div className="text-black/45 mb-1">Goal</div>
                  <div className="text-black font-medium">
                    {routeGoal ? `${routeGoal[0].toFixed(2)}, ${routeGoal[1].toFixed(2)}` : "none"}
                  </div>
                </div>
                <div className="rounded-2xl bg-[#F7F9FB] px-3 py-3">
                  <div className="text-black/45 mb-1">Active waypoint</div>
                  <div className="text-black font-medium">
                    {activeWaypoint ? `${activeWaypoint[0].toFixed(2)}, ${activeWaypoint[1].toFixed(2)}` : "none"}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
