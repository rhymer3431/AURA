import React, {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from "react";

import type {
  BootstrapData,
  DashboardContextValue,
  DashboardHistory,
  DashboardState,
  NumericSeries,
  SessionForm,
} from "./types";
import { createDashboardEventSource, requestJson } from "./network";

const DashboardContext = createContext<DashboardContextValue | null>(null);

export const DEFAULT_FORM: SessionForm = {
  plannerMode: "interactive",
  launchMode: "gui",
  scenePreset: "warehouse",
  viewerEnabled: true,
  memoryStore: true,
  detectionEnabled: true,
  locomotionConfig: {
    actionScale: "0.5",
    onnxDevice: "auto",
    cmdMaxVx: "0.5",
    cmdMaxVy: "0.3",
    cmdMaxWz: "0.8",
  },
  goalX: "2.0",
  goalY: "0.0",
};

type StateModel = {
  bootstrap: BootstrapData | null;
  state: DashboardState | null;
  history: DashboardHistory;
  error: string;
};

type Action =
  | { type: "bootstrap"; payload: BootstrapData }
  | { type: "state"; payload: DashboardState }
  | { type: "error"; payload: string };

function appendSeries(series: NumericSeries, value: number | null | undefined, timestamp: number): NumericSeries {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return series;
  }
  const next = [...series, { t: timestamp, v: value }];
  return next.slice(-20);
}

export function buildSessionPayload(form: SessionForm) {
  const payload: {
    plannerMode: SessionForm["plannerMode"];
    launchMode: SessionForm["launchMode"];
    scenePreset: string;
    viewerEnabled: boolean;
    memoryStore: boolean;
    detectionEnabled: boolean;
    locomotionConfig: {
      actionScale: number;
      onnxDevice: SessionForm["locomotionConfig"]["onnxDevice"];
      cmdMaxVx: number;
      cmdMaxVy: number;
      cmdMaxWz: number;
    };
  } = {
    plannerMode: form.plannerMode,
    launchMode: form.launchMode,
    scenePreset: form.scenePreset,
    viewerEnabled: form.viewerEnabled,
    memoryStore: form.memoryStore,
    detectionEnabled: form.detectionEnabled,
    locomotionConfig: {
      actionScale: Number(form.locomotionConfig.actionScale),
      onnxDevice: form.locomotionConfig.onnxDevice,
      cmdMaxVx: Number(form.locomotionConfig.cmdMaxVx),
      cmdMaxVy: Number(form.locomotionConfig.cmdMaxVy),
      cmdMaxWz: Number(form.locomotionConfig.cmdMaxWz),
    },
  };
  const locomotionValues = payload.locomotionConfig;
  if (
    !Number.isFinite(locomotionValues.actionScale) ||
    !Number.isFinite(locomotionValues.cmdMaxVx) ||
    !Number.isFinite(locomotionValues.cmdMaxVy) ||
    !Number.isFinite(locomotionValues.cmdMaxWz)
  ) {
    throw new Error("locomotion config must contain numeric values");
  }
  if (locomotionValues.actionScale <= 0) {
    throw new Error("locomotion actionScale must be positive");
  }
  if (locomotionValues.cmdMaxVx < 0 || locomotionValues.cmdMaxVy < 0) {
    throw new Error("locomotion cmdMaxVx and cmdMaxVy must be non-negative");
  }
  if (locomotionValues.cmdMaxWz <= 0) {
    throw new Error("locomotion cmdMaxWz must be positive");
  }
  return payload;
}

export function dashboardReducer(model: StateModel, action: Action): StateModel {
  if (action.type === "bootstrap") {
    return { ...model, bootstrap: action.payload, error: "" };
  }
  if (action.type === "error") {
    return { ...model, error: action.payload };
  }
  const timestamp = Math.round((action.payload.timestamp ?? Date.now() / 1000) as number);
  const navdpLatency = Number((action.payload.services.navdp?.latencyMs ?? NaN) || NaN);
  const dualLatency = Number((action.payload.services.dual?.latencyMs ?? NaN) || NaN);
  const stale = Number((action.payload.runtime.staleSec ?? NaN) || NaN);
  const goalDistance = Number((action.payload.runtime.goalDistanceM ?? NaN) || NaN);
  return {
    ...model,
    state: action.payload,
    error: "",
    history: {
      stale: appendSeries(model.history.stale, stale, timestamp),
      goalDistance: appendSeries(model.history.goalDistance, goalDistance, timestamp),
      navdpLatency: appendSeries(model.history.navdpLatency, navdpLatency, timestamp),
      dualLatency: appendSeries(model.history.dualLatency, dualLatency, timestamp),
    },
  };
}

export function DashboardProvider({ children }: { children: React.ReactNode }) {
  const [model, dispatch] = useReducer(dashboardReducer, {
    bootstrap: null,
    state: null,
    history: { stale: [], goalDistance: [], navdpLatency: [], dualLatency: [] },
    error: "",
  });
  const [form, setFormState] = useState<SessionForm>(DEFAULT_FORM);
  const [loading, setLoading] = useState(true);
  const eventSourceRef = useRef<EventSource | null>(null);

  async function refresh() {
    const nextState = await requestJson<DashboardState>("/api/state");
    dispatch({ type: "state", payload: nextState });
  }

  async function startSession() {
    setLoading(true);
    try {
      const payload = buildSessionPayload(form);
      const nextState = await requestJson<DashboardState>("/api/session/start", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      dispatch({ type: "state", payload: nextState });
    } catch (error) {
      dispatch({ type: "error", payload: error instanceof Error ? error.message : String(error) });
    } finally {
      setLoading(false);
    }
  }

  async function stopSession() {
    setLoading(true);
    try {
      const nextState = await requestJson<DashboardState>("/api/session/stop", {
        method: "POST",
        body: JSON.stringify({}),
      });
      dispatch({ type: "state", payload: nextState });
    } catch (error) {
      dispatch({ type: "error", payload: error instanceof Error ? error.message : String(error) });
    } finally {
      setLoading(false);
    }
  }

  async function submitTask(instruction: string) {
    const normalized = instruction.trim();
    if (normalized === "") {
      return;
    }
    try {
      await requestJson("/api/runtime/task", {
        method: "POST",
        body: JSON.stringify({ instruction: normalized }),
      });
      await refresh();
    } catch (error) {
      dispatch({ type: "error", payload: error instanceof Error ? error.message : String(error) });
    }
  }

  async function cancelTask() {
    try {
      await requestJson("/api/runtime/cancel", { method: "POST", body: JSON.stringify({}) });
      await refresh();
    } catch (error) {
      dispatch({ type: "error", payload: error instanceof Error ? error.message : String(error) });
    }
  }

  useEffect(() => {
    let mounted = true;
    async function bootstrap() {
      try {
        const [bootstrapData, stateData] = await Promise.all([
          requestJson<BootstrapData>("/api/bootstrap"),
          requestJson<DashboardState>("/api/state"),
        ]);
        if (!mounted) {
          return;
        }
        dispatch({ type: "bootstrap", payload: bootstrapData });
        dispatch({ type: "state", payload: stateData });
      } catch (error) {
        if (mounted) {
          dispatch({ type: "error", payload: error instanceof Error ? error.message : String(error) });
        }
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    }
    bootstrap();
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    const source = createDashboardEventSource("/api/events");
    eventSourceRef.current = source;
    source.addEventListener("state", (event) => {
      const nextState = JSON.parse((event as MessageEvent).data) as DashboardState;
      dispatch({ type: "state", payload: nextState });
    });
    source.onerror = () => {
      dispatch({ type: "error", payload: "실시간 이벤트 스트림 연결이 불안정합니다." });
    };
    return () => {
      source.close();
      eventSourceRef.current = null;
    };
  }, []);

  const value = useMemo<DashboardContextValue>(
    () => ({
      bootstrap: model.bootstrap,
      state: model.state,
      history: model.history,
      form,
      loading,
      error: model.error,
      setForm: (update) => setFormState((current) => ({ ...current, ...update })),
      startSession,
      stopSession,
      submitTask,
      cancelTask,
      refresh,
    }),
    [form, loading, model.bootstrap, model.error, model.history, model.state],
  );

  return <DashboardContext.Provider value={value}>{children}</DashboardContext.Provider>;
}

export function useDashboard() {
  const context = useContext(DashboardContext);
  if (context === null) {
    throw new Error("useDashboard must be used inside DashboardProvider");
  }
  return context;
}
