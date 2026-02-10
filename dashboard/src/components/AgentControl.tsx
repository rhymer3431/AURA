import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp, Eye, RotateCcw, RotateCw, Send } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { BACKEND_DISABLED, STREAM_BASE_URL } from '../application/streaming/streamConfig';

const SUPPORTED_ACTIONS = [
  'move_forward',
  'move_backward',
  'move_left',
  'move_right',
  'move_up',
  'move_down',
  'turn_left',
  'turn_right',
  'look_up',
  'look_down',
] as const;

type HabitatAction = (typeof SUPPORTED_ACTIONS)[number];

type ActionButtonProps = {
  action: HabitatAction;
  label: string;
  icon: ReactNode;
  onSend: (action: HabitatAction, repeat?: number) => Promise<void>;
  disabled: boolean;
};

function ActionButton({ action, label, icon, onSend, disabled }: ActionButtonProps) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={() => void onSend(action, 1)}
      className="inline-flex items-center justify-center gap-2 rounded-lg border border-[#e2e8f0] bg-white px-3 py-3 text-[13px] font-medium text-[#2d3748] transition-colors hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-60"
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}

export function AgentControl() {
  const baseUrl = useMemo(() => STREAM_BASE_URL.replace(/\/$/, ''), []);
  const [repeatCount, setRepeatCount] = useState('3');
  const [sending, setSending] = useState(false);
  const [lastAction, setLastAction] = useState<string>('-');
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const sendingRef = useRef(false);

  const sendAction = useCallback(async (action: HabitatAction, repeat = 1) => {
    if (BACKEND_DISABLED) {
      setError('백엔드 비활성 모드입니다. 백엔드를 실행한 뒤 다시 시도하세요.');
      return;
    }
    if (sendingRef.current) return;

    sendingRef.current = true;
    setSending(true);
    setError(null);
    setStatusMessage(null);

    try {
      const response = await fetch(`${baseUrl}/system/habitat/action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, repeat }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data?.message ?? `HTTP ${response.status}`);
      }

      setLastAction(repeat > 1 ? `${action} x${repeat}` : action);
      setStatusMessage('명령을 전송했습니다.');
    } catch (err) {
      const nextError = err instanceof Error ? err.message : 'Unknown error';
      setError(nextError === 'Failed to fetch' ? '백엔드 서버에 연결할 수 없습니다.' : nextError);
    } finally {
      sendingRef.current = false;
      setSending(false);
    }
  }, [baseUrl]);

  useEffect(() => {
    if (BACKEND_DISABLED) return;

    const keyToAction: Record<string, HabitatAction> = {
      w: 'move_forward',
      a: 'move_left',
      s: 'move_backward',
      d: 'move_right',
      z: 'move_down',
      x: 'move_up',
      q: 'turn_left',
      e: 'turn_right',
      i: 'look_up',
      k: 'look_down',
      arrowup: 'look_up',
      arrowdown: 'look_down',
      arrowleft: 'turn_left',
      arrowright: 'turn_right',
    };

    const pressedActions = new Set<HabitatAction>();
    let activeAction: HabitatAction | null = null;
    let repeatTimer: number | null = null;

    const clearRepeatTimer = () => {
      if (repeatTimer !== null) {
        window.clearInterval(repeatTimer);
        repeatTimer = null;
      }
    };

    const stopAction = (action: HabitatAction) => {
      pressedActions.delete(action);
      if (activeAction !== action) {
        return;
      }

      activeAction = null;
      const remaining = Array.from(pressedActions);
      if (remaining.length > 0) {
        activeAction = remaining[remaining.length - 1];
      } else {
        clearRepeatTimer();
      }
    };

    const startAction = (action: HabitatAction) => {
      activeAction = action;
      pressedActions.add(action);
      void sendAction(action, 1);

      if (repeatTimer === null) {
        repeatTimer = window.setInterval(() => {
          if (activeAction !== null) {
            void sendAction(activeAction, 1);
          }
        }, 120);
      }
    };

    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) {
        return;
      }

      const action = keyToAction[event.key.toLowerCase()];
      if (!action) return;

      event.preventDefault();
      if (activeAction === action) {
        return;
      }
      startAction(action);
    };

    const onKeyUp = (event: KeyboardEvent) => {
      const action = keyToAction[event.key.toLowerCase()];
      if (!action) return;

      event.preventDefault();
      stopAction(action);
    };

    const onBlur = () => {
      pressedActions.clear();
      activeAction = null;
      clearRepeatTimer();
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', onBlur);
    return () => {
      clearRepeatTimer();
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', onBlur);
    };
  }, [sendAction]);

  const parsedRepeat = Math.max(1, Math.min(10, Number.parseInt(repeatCount || '1', 10) || 1));

  return (
    <div className="flex flex-col space-y-6">
      {BACKEND_DISABLED && (
        <p className="text-[13px] text-[#d69e2e]">
          현재는 대시보드 전용 모드입니다. 백엔드 API 호출은 비활성화되어 있습니다.
        </p>
      )}

      <div className="grid min-h-0 grid-cols-[1.2fr_1fr] gap-6">
        <div className="panel flex min-h-0 flex-col">
          <h3 className="mb-4 text-[18px] text-[#2d3748]">Agent Teleop</h3>

          <div className="grid grid-cols-3 gap-3 text-[13px] text-[#2d3748]">
            <div />
            <ActionButton
              action="move_forward"
              label="Forward"
              icon={<ArrowUp className="size-4" />}
              onSend={sendAction}
              disabled={sending || BACKEND_DISABLED}
            />
            <div />
            <ActionButton
              action="move_left"
              label="Left"
              icon={<ArrowLeft className="size-4" />}
              onSend={sendAction}
              disabled={sending || BACKEND_DISABLED}
            />
            <ActionButton
              action="move_backward"
              label="Backward"
              icon={<ArrowDown className="size-4" />}
              onSend={sendAction}
              disabled={sending || BACKEND_DISABLED}
            />
            <ActionButton
              action="move_right"
              label="Right"
              icon={<ArrowRight className="size-4" />}
              onSend={sendAction}
              disabled={sending || BACKEND_DISABLED}
            />
          </div>

          <div className="mt-4 grid grid-cols-2 gap-3">
            <ActionButton
              action="turn_left"
              label="Turn Left"
              icon={<RotateCcw className="size-4" />}
              onSend={sendAction}
              disabled={sending || BACKEND_DISABLED}
            />
            <ActionButton
              action="turn_right"
              label="Turn Right"
              icon={<RotateCw className="size-4" />}
              onSend={sendAction}
              disabled={sending || BACKEND_DISABLED}
            />
            <ActionButton
              action="look_up"
              label="Look Up"
              icon={<Eye className="size-4" />}
              onSend={sendAction}
              disabled={sending || BACKEND_DISABLED}
            />
            <ActionButton
              action="look_down"
              label="Look Down"
              icon={<Eye className="size-4" />}
              onSend={sendAction}
              disabled={sending || BACKEND_DISABLED}
            />
            <ActionButton
              action="move_up"
              label="Move Up"
              icon={<ArrowUp className="size-4" />}
              onSend={sendAction}
              disabled={sending || BACKEND_DISABLED}
            />
            <ActionButton
              action="move_down"
              label="Move Down"
              icon={<ArrowDown className="size-4" />}
              onSend={sendAction}
              disabled={sending || BACKEND_DISABLED}
            />
          </div>

          <div className="mt-5 rounded-lg border border-[#e2e8f0] bg-white p-3">
            <div className="mb-2 text-[12px] uppercase tracking-wide text-[#a0aec0]">Repeat Action</div>
            <div className="flex items-center gap-3">
              <input
                type="number"
                min={1}
                max={10}
                value={repeatCount}
                onChange={(event) => setRepeatCount(event.target.value)}
                className="w-24 rounded-lg border border-[#e2e8f0] bg-white px-3 py-2 text-[13px]"
              />
              <button
                type="button"
                onClick={() => void sendAction('move_forward', parsedRepeat)}
                disabled={sending || BACKEND_DISABLED}
                className="inline-flex items-center gap-2 rounded-md border border-[#bee3f8] bg-[#ebf8ff] px-4 py-2 text-sm font-medium text-[#2b6cb0] transition-colors hover:bg-[#e2f2ff] disabled:cursor-not-allowed disabled:opacity-60"
              >
                <Send className="size-4" />
                Forward x{parsedRepeat}
              </button>
              <button
                type="button"
                onClick={() => void sendAction('turn_left', parsedRepeat)}
                disabled={sending || BACKEND_DISABLED}
                className="inline-flex items-center gap-2 rounded-md border border-[#bee3f8] bg-[#ebf8ff] px-4 py-2 text-sm font-medium text-[#2b6cb0] transition-colors hover:bg-[#e2f2ff] disabled:cursor-not-allowed disabled:opacity-60"
              >
                <Send className="size-4" />
                Turn Left x{parsedRepeat}
              </button>
            </div>
          </div>

          {statusMessage && <p className="mt-3 text-[13px] text-[#2f855a]">{statusMessage}</p>}
          {error && <p className="mt-3 text-[13px] text-[#c53030]">{error}</p>}
        </div>

        <div className="panel">
          <h3 className="mb-4 text-[18px] text-[#2d3748]">Control Status</h3>
          <div className="space-y-3 text-[13px] text-[#2d3748]">
            <div className="rounded-lg bg-gray-50 p-3">
              <div className="text-[11px] uppercase tracking-wide text-[#a0aec0]">Last Action</div>
              <div>{lastAction}</div>
            </div>
            <div className="rounded-lg bg-gray-50 p-3">
              <div className="text-[11px] uppercase tracking-wide text-[#a0aec0]">Keyboard</div>
              <div>W/A/S/D, Q/E, I/K, Arrow Keys</div>
            </div>
            <div className="rounded-lg bg-gray-50 p-3">
              <div className="text-[11px] uppercase tracking-wide text-[#a0aec0]">Action API</div>
              <div className="break-all">{baseUrl}/system/habitat/action</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
