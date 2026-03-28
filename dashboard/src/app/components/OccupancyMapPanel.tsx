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

export function OccupancyMapPanel() {
  const { state, form } = useDashboard();
  const [mapData, setMapData] = useState<OccupancyMapResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState("");

  const activeScenePreset = state?.session.config?.scenePreset ?? form.scenePreset;
  const runtime = asRecord(state?.runtime);
  const sensors = asRecord(state?.sensors);
  const selectedTarget = state?.selectedTargetSummary;

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
  const selectedTargetCanvas = mapWorldToCanvas(
    mapData,
    selectedTarget?.worldPose && selectedTarget.worldPose.length >= 2
      ? [selectedTarget.worldPose[0], selectedTarget.worldPose[1]]
      : null,
  );
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
    <div className="rounded-[24px] border border-[rgba(var(--ink-rgb),0.06)] bg-[var(--surface-1)] p-5">
      <div className="mb-4 flex items-center justify-between gap-4">
        <div>
          <div className="mb-1 flex items-center gap-2 text-[12px] text-[var(--text-tertiary)]">
            <Map className="size-4" />
            Occupancy Localization
          </div>
          <h3 className="text-[18px] font-semibold text-[var(--foreground)]">Occupancy Map</h3>
          <p className="mt-1 text-[11px] text-[var(--text-tertiary)]">
            scene: <span className="text-[var(--text-secondary)]">{activeScenePreset || "inactive"}</span>
          </p>
        </div>
        <div className="grid min-w-[240px] grid-cols-2 gap-2.5">
          <div className="rounded-[18px] border border-[rgba(var(--ink-rgb),0.05)] bg-[var(--surface-strong)] px-3.5 py-3">
            <div className="mb-1 text-[11px] text-[var(--text-tertiary)]">Robot XY</div>
            <div className="text-[14px] font-medium text-[var(--foreground)]">
              {formatCoord(robotPose[0])}, {formatCoord(robotPose[1])}
            </div>
          </div>
          <div className="rounded-[18px] border border-[rgba(var(--ink-rgb),0.05)] bg-[var(--surface-strong)] px-3.5 py-3">
            <div className="mb-1 text-[11px] text-[var(--text-tertiary)]">Global Route</div>
            <div className="text-[14px] font-medium text-[var(--foreground)]">
              {routeStatus} {routeEnabled ? `(${Number(runtime.globalRouteWaypointIndex ?? 0)}/${Number(runtime.globalRouteWaypointCount ?? 0)})` : ""}
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
        <div className="xl:col-span-8">
          <div className="overflow-hidden rounded-[20px] border border-[rgba(var(--ink-rgb),0.08)] bg-[var(--surface-strong)]">
            <div className="flex items-center justify-between border-b border-[rgba(var(--ink-rgb),0.06)] bg-[var(--surface-2)] px-3.5 py-2.5">
              <div className="flex items-center gap-2 text-[12px] text-[var(--text-secondary)]">
                <Route className="size-4" />
                {mapData?.label ?? "Occupancy source"}
              </div>
              <div className="text-[11px] text-[var(--text-tertiary)]">
                res {typeof mapData?.resolutionMpp === "number" ? `${mapData.resolutionMpp.toFixed(3)} m/px` : "n/a"}
              </div>
            </div>

            {loading && <div className="px-4 py-8 text-[12px] text-[var(--text-secondary)]">occupancy map metadata를 불러오는 중입니다.</div>}

            {!loading && (loadError !== "" || mapData?.available === false || mapData === null) && (
              <div className="px-4 py-8 text-[12px] text-[var(--text-secondary)]">
                <div className="mb-2 font-medium text-[var(--foreground)]">이 scene에는 occupancy map 표시를 준비하지 못했습니다.</div>
                <div>{loadError || mapData?.reason || "선택된 scene preset에 대응하는 occupancy 자산이 없습니다."}</div>
              </div>
            )}

            {!loading && mapData?.available === true && (
              <div className="p-3.5">
                <div
                  className="relative w-full overflow-hidden rounded-[18px] border border-[rgba(var(--ink-rgb),0.1)] bg-[var(--surface-2)]"
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
                        stroke="var(--signal-amber)"
                        strokeWidth={3}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        filter="url(#routeGlow)"
                      />
                    )}
                    {goalCanvas !== null && (
                      <g transform={`translate(${goalCanvas.x},${goalCanvas.y})`}>
                        <rect x={-7} y={-7} width={14} height={14} transform="rotate(45)" fill="var(--foreground)" stroke="var(--surface-strong)" strokeWidth={2} />
                      </g>
                    )}
                    {activeWaypointCanvas !== null && (
                      <g transform={`translate(${activeWaypointCanvas.x},${activeWaypointCanvas.y})`}>
                        <circle r={8} fill="var(--tone-cyan-border)" />
                        <circle r={4} fill="var(--signal-cyan)" stroke="var(--surface-strong)" strokeWidth={1.5} />
                      </g>
                    )}
                    {selectedTargetCanvas !== null && (
                      <g transform={`translate(${selectedTargetCanvas.x},${selectedTargetCanvas.y})`}>
                        <circle r={10} fill="none" stroke="var(--signal-coral)" strokeWidth={2.5} />
                        <circle r={3} fill="var(--signal-coral)" />
                      </g>
                    )}
                    {robotCanvas !== null && (
                      <g transform={`translate(${robotCanvas.x},${robotCanvas.y}) rotate(${headingDeg})`}>
                        <circle r={12} fill="var(--tone-emerald-border)" />
                        <path d="M 11 0 L -7 6 L -3 0 L -7 -6 Z" fill="var(--signal-emerald)" stroke="var(--surface-strong)" strokeWidth={1.5} />
                      </g>
                    )}
                  </svg>
                </div>

                <div className="mt-3 flex flex-wrap items-center gap-2.5 text-[11px] text-[var(--text-secondary)]">
                  <div className="inline-flex items-center gap-2 rounded-full bg-[var(--tone-amber-bg)] px-3 py-1.5">
                    <span className="size-2 rounded-full bg-[var(--signal-amber)]" />
                    route
                  </div>
                  <div className="inline-flex items-center gap-2 rounded-full bg-[var(--tone-cyan-bg)] px-3 py-1.5">
                    <span className="size-2 rounded-full bg-[var(--signal-cyan)]" />
                    active waypoint
                  </div>
                  <div className="inline-flex items-center gap-2 rounded-full bg-[var(--tone-emerald-bg)] px-3 py-1.5">
                    <span className="size-2 rounded-full bg-[var(--signal-emerald)]" />
                    robot
                  </div>
                  <div className="inline-flex items-center gap-2 rounded-full bg-[var(--tone-slate-bg)] px-3 py-1.5">
                    <span className="size-2 rotate-45 bg-[var(--foreground)]" />
                    final goal
                  </div>
                  {selectedTargetCanvas !== null ? (
                    <div className="inline-flex items-center gap-2 rounded-full bg-[var(--tone-coral-bg)] px-3 py-1.5">
                      <span className="size-2 rounded-full bg-[var(--signal-coral)]" />
                      selected target
                    </div>
                  ) : null}
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="space-y-3 xl:col-span-4">
          <div className="rounded-[20px] border border-[rgba(var(--ink-rgb),0.06)] bg-[var(--surface-strong)] p-3.5">
            <div className="mb-3 flex items-center gap-2 text-[12px] text-[var(--text-tertiary)]">
              <Crosshair className="size-4" />
              Localization Snapshot
            </div>
            <div className="grid grid-cols-2 gap-2.5 text-[12px]">
              <div className="rounded-[16px] bg-[var(--surface-2)] px-3 py-2.5">
                <div className="mb-1 text-[var(--text-tertiary)]">Robot X</div>
                <div className="font-medium text-[var(--foreground)]">{formatCoord(robotPose[0])}</div>
              </div>
              <div className="rounded-[16px] bg-[var(--surface-2)] px-3 py-2.5">
                <div className="mb-1 text-[var(--text-tertiary)]">Robot Y</div>
                <div className="font-medium text-[var(--foreground)]">{formatCoord(robotPose[1])}</div>
              </div>
              <div className="rounded-[16px] bg-[var(--surface-2)] px-3 py-2.5">
                <div className="mb-1 text-[var(--text-tertiary)]">Yaw</div>
                <div className="font-medium text-[var(--foreground)]">{robotYawRad.toFixed(2)} rad</div>
              </div>
              <div className="rounded-[16px] bg-[var(--surface-2)] px-3 py-2.5">
                <div className="mb-1 text-[var(--text-tertiary)]">Goal Dist</div>
                <div className="font-medium text-[var(--foreground)]">
                  {typeof runtime.goalDistanceM === "number" ? `${Number(runtime.goalDistanceM).toFixed(2)}m` : "n/a"}
                </div>
              </div>
            </div>
          </div>

          <div className="rounded-[20px] border border-[rgba(var(--ink-rgb),0.06)] bg-[var(--surface-strong)] p-3.5">
            <div className="mb-3 flex items-center gap-2 text-[12px] text-[var(--text-tertiary)]">
              <Target className="size-4" />
              Route Debug
            </div>
            <div className="space-y-2.5 text-[12px]">
              <div className="rounded-[16px] bg-[var(--surface-2)] px-3 py-2.5">
                <div className="mb-1 text-[var(--text-tertiary)]">Replan reason</div>
                <div className="font-medium text-[var(--foreground)]">{stringValue(runtime.globalRouteLastReplanReason, "none") || "none"}</div>
              </div>
              <div className="rounded-[16px] bg-[var(--surface-2)] px-3 py-2.5">
                <div className="mb-1 text-[var(--text-tertiary)]">Route error</div>
                <div className="break-words font-medium text-[var(--foreground)]">{stringValue(runtime.globalRouteLastError, "clear") || "clear"}</div>
              </div>
              <div className="rounded-[16px] bg-[var(--surface-2)] px-3 py-2.5">
                <div className="mb-1 text-[var(--text-tertiary)]">Map bounds</div>
                <div className="font-medium text-[var(--foreground)]">
                  x [{formatCoord(mapData?.xMin)}, {formatCoord(mapData?.xMax)}]
                </div>
                <div className="font-medium text-[var(--foreground)]">
                  y [{formatCoord(mapData?.yMin)}, {formatCoord(mapData?.yMax)}]
                </div>
              </div>
              <div className="rounded-[16px] bg-[var(--surface-2)] px-3 py-2.5">
                <div className="mb-1 text-[var(--text-tertiary)]">Active waypoint</div>
                <div className="font-medium text-[var(--foreground)]">
                  {activeWaypoint ? `${activeWaypoint[0].toFixed(2)}, ${activeWaypoint[1].toFixed(2)}` : "none"}
                </div>
              </div>
              <div className="rounded-[16px] bg-[var(--surface-2)] px-3 py-2.5">
                <div className="mb-1 text-[var(--text-tertiary)]">Selected target</div>
                <div className="font-medium text-[var(--foreground)]">
                  {selectedTarget?.worldPose ? `${selectedTarget.worldPose[0].toFixed(2)}, ${selectedTarget.worldPose[1].toFixed(2)}` : "none"}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
