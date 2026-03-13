import { render, screen, waitFor } from "@testing-library/react";
import { vi } from "vitest";

import { useWebRTCViewer } from "./useWebRTCViewer";

class FakeDataChannel {
  label: string;
  readyState = "open";
  onmessage: ((event: { data: string }) => void) | null = null;

  constructor(label: string) {
    this.label = label;
  }

  send() {}
}

class FakePeerConnection {
  static instance: FakePeerConnection | null = null;

  iceGatheringState = "new";
  connectionState = "new";
  localDescription: RTCSessionDescriptionInit | null = null;
  remoteDescription: RTCSessionDescriptionInit | null = null;
  transceivers: Array<{ kind: string; direction: string }> = [];
  channels: FakeDataChannel[] = [];
  listeners: Record<string, Array<() => void>> = {};
  ontrack: ((event: { track: MediaStreamTrack }) => void) | null = null;
  onconnectionstatechange: (() => void) | null = null;

  constructor() {
    FakePeerConnection.instance = this;
  }

  createDataChannel(label: string) {
    const channel = new FakeDataChannel(label);
    this.channels.push(channel);
    return channel;
  }

  addTransceiver(kind: string, options: { direction: string }) {
    this.transceivers.push({ kind, direction: options.direction });
  }

  addEventListener(type: string, callback: () => void) {
    this.listeners[type] = [...(this.listeners[type] ?? []), callback];
  }

  removeEventListener(type: string, callback: () => void) {
    this.listeners[type] = (this.listeners[type] ?? []).filter((item) => item !== callback);
  }

  async createOffer() {
    return { sdp: "local-offer", type: "offer" as const };
  }

  async setLocalDescription(description: RTCSessionDescriptionInit) {
    this.localDescription = description;
    this.iceGatheringState = "complete";
    (this.listeners.icegatheringstatechange ?? []).forEach((callback) => callback());
  }

  async setRemoteDescription(description: RTCSessionDescriptionInit) {
    this.remoteDescription = description;
    this.connectionState = "connected";
    this.onconnectionstatechange?.();
    this.channels.find((item) => item.label === "state")?.onmessage?.({
      data: JSON.stringify({ type: "session_ready", trackRoles: ["rgb", "depth"] }),
    });
    this.channels.find((item) => item.label === "state")?.onmessage?.({
      data: JSON.stringify({ type: "snapshot", image: { width: 320, height: 180 } }),
    });
    this.channels.find((item) => item.label === "telemetry")?.onmessage?.({
      data: JSON.stringify({ type: "frame_meta", detections: [], trajectoryPixels: [] }),
    });
  }

  close() {
    this.connectionState = "closed";
  }
}

function HookProbe() {
  const viewer = useWebRTCViewer({ basePath: "/api/webrtc", enabled: true });

  return (
    <div>
      <div data-testid="connected">{String(viewer.connected)}</div>
      <div data-testid="tracks">{viewer.trackRoles.join(",")}</div>
      <div data-testid="snapshot">{String(viewer.snapshotRef.current?.type ?? "")}</div>
      <div data-testid="telemetry">{String(viewer.telemetryRef.current?.type ?? "")}</div>
    </div>
  );
}

describe("useWebRTCViewer", () => {
  it("creates recvonly transceivers and completes offer flow", async () => {
    Object.defineProperty(globalThis, "RTCPeerConnection", {
      value: FakePeerConnection,
      writable: true,
    });
    Object.defineProperty(globalThis, "MediaStream", {
      value: class {
        constructor(_tracks: unknown[]) {}
      },
      writable: true,
    });
    Object.defineProperty(globalThis, "fetch", {
      value: vi
        .fn()
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ iceServers: [], enableDepthTrack: true }),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ sdp: "answer", type: "answer" }),
        }),
      writable: true,
    });

    render(<HookProbe />);

    await waitFor(() => {
      expect(screen.getByTestId("connected")).toHaveTextContent("true");
      expect(screen.getByTestId("tracks")).toHaveTextContent("rgb,depth");
      expect(screen.getByTestId("snapshot")).toHaveTextContent("snapshot");
      expect(screen.getByTestId("telemetry")).toHaveTextContent("frame_meta");
    });

    expect(FakePeerConnection.instance?.transceivers).toHaveLength(2);
    expect(globalThis.fetch).toHaveBeenNthCalledWith(
      1,
      "/api/webrtc/config",
      {
        headers: {
          "Content-Type": "application/json",
        },
      },
    );
    expect(globalThis.fetch).toHaveBeenNthCalledWith(
      2,
      "/api/webrtc/offer",
      expect.objectContaining({
        method: "POST",
      }),
    );
  });
});
