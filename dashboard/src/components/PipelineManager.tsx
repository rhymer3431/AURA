import { Play, RefreshCw, Square } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { BACKEND_DISABLED, STREAM_BASE_URL } from '../application/streaming/streamConfig';

type PipelineStatus = {
  running: boolean;
  pid: number | null;
  startedAt: number | null;
  uptimeSec: number | null;
  lastExitCode: number | null;
  logPath: string | null;
};

type PipelineOptions = {
  enableSemantic: boolean;
  enableOctomap: boolean;
  enableLlm: boolean;
  useNav2: boolean;
  nav2AutoExplore: boolean;
  habitatManualOnly: boolean;
  habitatAddPlayerAgent: boolean;
  habitatDataset: string;
  octomapResolution: string;
  yoloModelPath: string;
};

type HabitatDatasetOption = {
  label: string;
  value: string;
};

const habitatDatasetOptions: HabitatDatasetOption[] = [
  {
    label: 'ReplicaCAD (apt_1)',
    value: '/home/mangoo/project/habitat-sim/data/replica_cad/replicaCAD.scene_dataset_config.json',
  },
  {
    label: 'ReplicaCAD Custom (multiroom_compound)',
    value: '/home/mangoo/project/habitat-sim/data/replica_cad_custom/replicaCAD_custom.scene_dataset_config.json',
  },
  {
    label: 'Modern Apartment',
    value:
      '/home/mangoo/project/habitat-sim/data/scene_datasets/modern_apartment/modern_apartment.scene_dataset_config.json',
  },
];

type ToggleOptionKey =
  | 'enableSemantic'
  | 'enableOctomap'
  | 'enableLlm'
  | 'useNav2'
  | 'nav2AutoExplore'
  | 'habitatManualOnly'
  | 'habitatAddPlayerAgent';

const defaultStatus: PipelineStatus = {
  running: false,
  pid: null,
  startedAt: null,
  uptimeSec: null,
  lastExitCode: null,
  logPath: null,
};

const defaultOptions: PipelineOptions = {
  enableSemantic: true,
  enableOctomap: true,
  enableLlm: false,
  useNav2: true,
  nav2AutoExplore: true,
  habitatManualOnly: false,
  habitatAddPlayerAgent: false,
  habitatDataset: habitatDatasetOptions[0].value,
  octomapResolution: '0.15',
  yoloModelPath: '',
};

const formatUptime = (uptimeSec: number | null) => {
  if (uptimeSec === null || Number.isNaN(uptimeSec)) return '-';
  const sec = Math.max(0, Math.floor(uptimeSec));
  const hh = Math.floor(sec / 3600);
  const mm = Math.floor((sec % 3600) / 60);
  const ss = sec % 60;
  return `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
};

export function PipelineManager() {
  const baseUrl = useMemo(() => STREAM_BASE_URL.replace(/\/$/, ''), []);
  const [status, setStatus] = useState<PipelineStatus>(defaultStatus);
  const [options, setOptions] = useState<PipelineOptions>(defaultOptions);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refreshStatus = async () => {
    if (BACKEND_DISABLED) {
      setError(null);
      return;
    }
    try {
      const response = await fetch(`${baseUrl}/system/pipeline/status`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.message ?? `HTTP ${response.status}`);
      }
      setStatus((data?.pipeline as PipelineStatus | undefined) ?? defaultStatus);
      setError(null);
    } catch (err) {
      const nextError = err instanceof Error ? err.message : 'Unknown error';
      setError(nextError === 'Failed to fetch' ? '백엔드 서버에 연결할 수 없습니다.' : nextError);
    }
  };

  useEffect(() => {
    if (BACKEND_DISABLED) return;
    void refreshStatus();
    const timer = window.setInterval(() => {
      void refreshStatus();
    }, 3000);
    return () => window.clearInterval(timer);
  }, []);

  const handleOptionToggle = (key: ToggleOptionKey) => {
    setOptions((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  const handleStart = async () => {
    if (BACKEND_DISABLED) {
      setError('백엔드 비활성 모드입니다. 백엔드 실행 후 다시 시도하세요.');
      return;
    }
    if (loading) return;
    setLoading(true);
    setMessage(null);
    setError(null);
    try {
      const payload = {
        ...options,
        octomapResolution: Number(options.octomapResolution),
      };
      const response = await fetch(`${baseUrl}/system/pipeline/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data?.message ?? `HTTP ${response.status}`);
      }
      setMessage('파이프라인 실행 요청을 보냈습니다.');
      await refreshStatus();
    } catch (err) {
      const nextError = err instanceof Error ? err.message : 'Unknown error';
      setError(nextError === 'Failed to fetch' ? '백엔드 서버에 연결할 수 없습니다.' : nextError);
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    if (BACKEND_DISABLED) {
      setError('백엔드 비활성 모드입니다. 백엔드 실행 후 다시 시도하세요.');
      return;
    }
    if (loading) return;
    const confirmed = window.confirm('실행 중인 파이프라인을 종료할까요?');
    if (!confirmed) return;

    setLoading(true);
    setMessage(null);
    setError(null);
    try {
      const response = await fetch(`${baseUrl}/system/shutdown`, { method: 'POST' });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data?.message ?? `HTTP ${response.status}`);
      }
      setMessage('종료 신호를 전송했습니다.');
      await refreshStatus();
    } catch (err) {
      const nextError = err instanceof Error ? err.message : 'Unknown error';
      setError(nextError === 'Failed to fetch' ? '백엔드 서버에 연결할 수 없습니다.' : nextError);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-full flex-col space-y-6 pt-4">
      <p className="text-[14px] text-[#a0aec0]">
        Dashboard에서 파이프라인 실행 옵션을 선택하고 시작/종료를 제어합니다.
      </p>
      {BACKEND_DISABLED && (
        <p className="text-[13px] text-[#d69e2e]">
          현재는 대시보드 전용 모드입니다. 백엔드 API 호출은 비활성화되어 있습니다.
        </p>
      )}

      <div className="grid h-full min-h-0 grid-cols-[1.4fr_1fr] gap-6">
        <div className="panel flex min-h-0 flex-col">
          <h3 className="mb-4 text-[18px] text-[#2d3748]">Pipeline Options</h3>

          <div className="grid grid-cols-2 gap-3 text-[14px] text-[#2d3748]">
            <label className="flex items-center gap-2 rounded-lg border border-[#e2e8f0] bg-white px-3 py-2">
              <input
                type="checkbox"
                checked={options.enableSemantic}
                onChange={() => handleOptionToggle('enableSemantic')}
              />
              Enable Semantic
            </label>
            <label className="flex items-center gap-2 rounded-lg border border-[#e2e8f0] bg-white px-3 py-2">
              <input
                type="checkbox"
                checked={options.enableOctomap}
                onChange={() => handleOptionToggle('enableOctomap')}
              />
              Enable Octomap
            </label>
            <label className="flex items-center gap-2 rounded-lg border border-[#e2e8f0] bg-white px-3 py-2">
              <input
                type="checkbox"
                checked={options.enableLlm}
                onChange={() => handleOptionToggle('enableLlm')}
              />
              Enable LLM
            </label>
            <label className="flex items-center gap-2 rounded-lg border border-[#e2e8f0] bg-white px-3 py-2">
              <input
                type="checkbox"
                checked={options.useNav2}
                onChange={() => handleOptionToggle('useNav2')}
              />
              Use Nav2
            </label>
            <label className="flex items-center gap-2 rounded-lg border border-[#e2e8f0] bg-white px-3 py-2">
              <input
                type="checkbox"
                checked={options.nav2AutoExplore}
                onChange={() => handleOptionToggle('nav2AutoExplore')}
              />
              Nav2 Auto Explore
            </label>
            <label className="flex items-center gap-2 rounded-lg border border-[#e2e8f0] bg-white px-3 py-2">
              <input
                type="checkbox"
                checked={options.habitatManualOnly}
                onChange={() => handleOptionToggle('habitatManualOnly')}
              />
              Habitat Manual Only
            </label>
            <label className="flex items-center gap-2 rounded-lg border border-[#e2e8f0] bg-white px-3 py-2">
              <input
                type="checkbox"
                checked={options.habitatAddPlayerAgent}
                onChange={() => handleOptionToggle('habitatAddPlayerAgent')}
              />
              Habitat Add Player Agent
            </label>
          </div>

          <div className="mt-4 grid grid-cols-1 gap-3 text-[14px] text-[#2d3748]">
            <label className="flex flex-col gap-1">
              <span className="text-[12px] uppercase tracking-wide text-[#a0aec0]">
                Habitat Dataset
              </span>
              <select
                value={options.habitatDataset}
                onChange={(event) =>
                  setOptions((prev) => ({
                    ...prev,
                    habitatDataset: event.target.value,
                  }))
                }
                className="rounded-lg border border-[#e2e8f0] bg-white px-3 py-2"
              >
                {habitatDatasetOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="flex flex-col gap-1">
              <span className="text-[12px] uppercase tracking-wide text-[#a0aec0]">
                Octomap Resolution
              </span>
              <input
                type="number"
                step="0.01"
                value={options.octomapResolution}
                onChange={(event) =>
                  setOptions((prev) => ({
                    ...prev,
                    octomapResolution: event.target.value,
                  }))
                }
                className="rounded-lg border border-[#e2e8f0] bg-white px-3 py-2"
              />
            </label>

            <label className="flex flex-col gap-1">
              <span className="text-[12px] uppercase tracking-wide text-[#a0aec0]">
                YOLO Model Path (optional)
              </span>
              <input
                type="text"
                value={options.yoloModelPath}
                onChange={(event) =>
                  setOptions((prev) => ({
                    ...prev,
                    yoloModelPath: event.target.value,
                  }))
                }
                placeholder="/home/mangoo/project/weights/yoloe-26s-seg.pt"
                className="rounded-lg border border-[#e2e8f0] bg-white px-3 py-2"
              />
            </label>
          </div>

          <div className="mt-5 flex items-center gap-3">
            <button
              type="button"
              onClick={handleStart}
              disabled={BACKEND_DISABLED || loading || status.running}
              className="inline-flex items-center gap-2 rounded-md border border-[#c6f6d5] bg-[#f0fff4] px-4 py-2 text-sm font-medium text-[#2f855a] transition-colors hover:bg-[#e6ffed] disabled:cursor-not-allowed disabled:opacity-70"
            >
              <Play className="size-4" />
              {status.running ? '실행 중' : '실행'}
            </button>
            <button
              type="button"
              onClick={handleStop}
              disabled={BACKEND_DISABLED || loading || !status.running}
              className="inline-flex items-center gap-2 rounded-md border border-[#fed7d7] bg-[#fff5f5] px-4 py-2 text-sm font-medium text-[#c53030] transition-colors hover:bg-[#fff0f0] disabled:cursor-not-allowed disabled:opacity-70"
            >
              <Square className="size-4" />
              종료
            </button>
            <button
              type="button"
              onClick={() => void refreshStatus()}
              disabled={BACKEND_DISABLED || loading}
              className="inline-flex items-center gap-2 rounded-md border border-[#e2e8f0] bg-white px-4 py-2 text-sm font-medium text-[#2d3748] transition-colors hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-70"
            >
              <RefreshCw className="size-4" />
              상태 새로고침
            </button>
          </div>

          {message && <p className="mt-3 text-[13px] text-[#2f855a]">{message}</p>}
          {error && <p className="mt-3 text-[13px] text-[#c53030]">{error}</p>}
        </div>

        <div className="panel">
          <h3 className="mb-4 text-[18px] text-[#2d3748]">Pipeline Status</h3>
          <div className="space-y-3 text-[13px] text-[#2d3748]">
            <div className="rounded-lg bg-gray-50 p-3">
              <div className="text-[11px] uppercase tracking-wide text-[#a0aec0]">State</div>
              <div>{status.running ? 'Running' : 'Stopped'}</div>
            </div>
            <div className="rounded-lg bg-gray-50 p-3">
              <div className="text-[11px] uppercase tracking-wide text-[#a0aec0]">PID</div>
              <div>{status.pid ?? '-'}</div>
            </div>
            <div className="rounded-lg bg-gray-50 p-3">
              <div className="text-[11px] uppercase tracking-wide text-[#a0aec0]">Uptime</div>
              <div>{formatUptime(status.uptimeSec)}</div>
            </div>
            <div className="rounded-lg bg-gray-50 p-3">
              <div className="text-[11px] uppercase tracking-wide text-[#a0aec0]">Last Exit Code</div>
              <div>{status.lastExitCode ?? '-'}</div>
            </div>
            <div className="rounded-lg bg-gray-50 p-3">
              <div className="text-[11px] uppercase tracking-wide text-[#a0aec0]">Log Path</div>
              <div className="break-all">{status.logPath ?? '-'}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
