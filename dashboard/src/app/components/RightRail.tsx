import { motion } from "motion/react";
import { BellRing, Clock3, Radar, Waypoints } from "lucide-react";

import { useDashboard } from "../state";
import { asRecord, formatMs, recentLogs, statusLabel, statusTone, stringValue } from "../selectors";
import { ConsoleBadge } from "./console-ui";
import { cn } from "./ui/utils";

function formatLogTime(timestampNs: number | undefined) {
  if (typeof timestampNs !== "number" || !Number.isFinite(timestampNs)) {
    return "just now";
  }
  const date = new Date(Math.round(timestampNs / 1_000_000));
  if (Number.isNaN(date.getTime())) {
    return "just now";
  }
  return date.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
  }).toLowerCase();
}

const statusDotClass: Record<string, string> = {
  green: "bg-[var(--signal-emerald)]",
  amber: "bg-[var(--signal-amber)]",
  red: "bg-[var(--signal-coral)]",
  slate: "bg-[rgba(0,0,0,0.16)]",
};

export function RightRail({
  className,
  mobile = false,
}: {
  className?: string;
  mobile?: boolean;
}) {
  const { state } = useDashboard();
  const runtime = asRecord(state?.runtime);
  const transport = asRecord(state?.transport);
  const logs = recentLogs(state, 4);
  const watchlist = (state?.processes ?? [])
    .filter((item) => item.required)
    .slice(0, 4);
  const runtimeModes = asRecord(runtime.modes);

  return (
    <aside
      className={cn("dashboard-right-rail", className)}
      data-mobile={mobile ? "true" : "false"}
      aria-label="Live context rail"
    >
      <motion.section
        className="dashboard-rail-block"
        initial={{ opacity: 0, x: 18 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      >
        <div className="dashboard-rail-heading">
          <BellRing className="size-4" />
          <h2 className="dashboard-rail-title">Notifications</h2>
        </div>
        <div className="space-y-3">
          {logs.length === 0 ? (
            <div className="rounded-[16px] bg-[var(--surface-2)] px-3.5 py-3.5 text-[11px] text-[var(--text-tertiary)]">
              No recent runtime events yet.
            </div>
          ) : (
            logs.map((log, index) => (
              <motion.div
                key={`${log.source}-${index}`}
                className="dashboard-rail-item"
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.28, delay: 0.08 + index * 0.05 }}
              >
                <div className="flex items-start gap-3">
                  <span className="mt-1 size-2 rounded-full bg-[var(--signal-cyan)]" />
                  <div className="min-w-0">
                    <div className="truncate text-[13px] font-semibold text-[var(--foreground)]">
                      {stringValue(log.source, "runtime")}
                    </div>
                    <div className="mt-1 line-clamp-2 text-[11px] leading-5 text-[var(--text-secondary)]">
                      {stringValue(log.message, "No message")}
                    </div>
                    <div className="mt-1.5 text-[11px] text-[var(--text-tertiary)]">
                      {formatLogTime(log.timestampNs)}
                    </div>
                  </div>
                </div>
              </motion.div>
            ))
          )}
        </div>
      </motion.section>

      <motion.section
        className="dashboard-rail-block"
        initial={{ opacity: 0, x: 18 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.4, delay: 0.06, ease: [0.22, 1, 0.36, 1] }}
      >
        <div className="dashboard-rail-heading">
          <Radar className="size-4" />
          <h2 className="dashboard-rail-title">Live context</h2>
        </div>

        <div className="rounded-[16px] bg-[var(--tone-violet-bg)] px-3.5 py-3">
          <div className="flex items-center justify-between gap-3">
            <div>
              <div className="dashboard-eyebrow mb-1">Session</div>
              <div className="text-[15px] font-semibold text-[var(--foreground)]">
                {state?.session.active ? "Live" : "Idle"}
              </div>
            </div>
            <ConsoleBadge tone={state?.session.active ? "emerald" : "amber"}>
              {state?.session.active ? "connected" : "standby"}
            </ConsoleBadge>
          </div>
        </div>

        <div className="mt-2.5 space-y-2">
          <div className="dashboard-rail-item">
            <div className="dashboard-eyebrow">Execution mode</div>
            <div className="mt-1 text-[13px] font-semibold text-[var(--foreground)]">
              {String(runtime.executionMode ?? runtimeModes.executionMode ?? "IDLE")}
            </div>
          </div>
          <div className="dashboard-rail-item">
            <div className="dashboard-eyebrow">Frame freshness</div>
            <div className="mt-1 flex items-center justify-between gap-3">
              <span className="text-[13px] font-semibold text-[var(--foreground)]">
                {formatMs(transport.frameAgeMs, "n/a")}
              </span>
              <Clock3 className="size-4 text-[var(--text-tertiary)]" />
            </div>
          </div>
          <div className="dashboard-rail-item">
            <div className="dashboard-eyebrow">Source</div>
            <div className="mt-1 text-[13px] font-semibold text-[var(--foreground)]">
              {stringValue(asRecord(state?.sensors).source, "aura_runtime")}
            </div>
          </div>
        </div>
      </motion.section>

      <motion.section
        className="dashboard-rail-block"
        initial={{ opacity: 0, x: 18 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.4, delay: 0.12, ease: [0.22, 1, 0.36, 1] }}
      >
        <div className="dashboard-rail-heading">
          <Waypoints className="size-4" />
          <h2 className="dashboard-rail-title">Watchlist</h2>
        </div>

        <div className="space-y-2.5">
          {watchlist.length === 0 ? (
            <div className="dashboard-rail-item text-[11px] text-[var(--text-tertiary)]">
              Required process telemetry is not available.
            </div>
          ) : (
            watchlist.map((item, index) => {
              const tone = statusTone(item.state);
              return (
                <motion.div
                  key={item.name}
                  className="dashboard-rail-item"
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.24, delay: 0.16 + index * 0.04 }}
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className={`size-2 rounded-full ${statusDotClass[tone] ?? statusDotClass.slate}`} />
                        <div className="truncate text-[13px] font-semibold text-[var(--foreground)]">
                          {item.name}
                        </div>
                      </div>
                      <div className="mt-1 text-[11px] text-[var(--text-tertiary)]">
                        PID {item.pid ?? "n/a"}
                      </div>
                    </div>
                    <div className="text-[11px] text-[var(--text-secondary)]">
                      {statusLabel(item.state)}
                    </div>
                  </div>
                </motion.div>
              );
            })
          )}
        </div>
      </motion.section>
    </aside>
  );
}
