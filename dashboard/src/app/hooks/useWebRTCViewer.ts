import { useEffect, useRef, useState } from "react";

import type { ViewerStateMessage, ViewerTelemetryMessage } from "../types";

type ViewerState = {
  connected: boolean;
  error: string;
  session: ViewerStateMessage | null;
  snapshot: ViewerStateMessage | null;
  telemetry: ViewerTelemetryMessage | null;
  trackRoles: string[];
};

type ViewerOptions = {
  basePath: string;
  enabled: boolean;
};

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return (await response.json()) as T;
}

async function waitForIceGatheringComplete(peer: RTCPeerConnection): Promise<void> {
  if (peer.iceGatheringState === "complete") {
    return;
  }
  await new Promise<void>((resolve) => {
    const handleChange = () => {
      if (peer.iceGatheringState === "complete") {
        peer.removeEventListener("icegatheringstatechange", handleChange);
        resolve();
      }
    };
    peer.addEventListener("icegatheringstatechange", handleChange);
    window.setTimeout(() => {
      peer.removeEventListener("icegatheringstatechange", handleChange);
      resolve();
    }, 10_000);
  });
}

export function useWebRTCViewer({ basePath, enabled }: ViewerOptions) {
  const rgbVideoRef = useRef<HTMLVideoElement | null>(null);
  const depthVideoRef = useRef<HTMLVideoElement | null>(null);
  const peerRef = useRef<RTCPeerConnection | null>(null);
  const [state, setState] = useState<ViewerState>({
    connected: false,
    error: "",
    session: null,
    snapshot: null,
    telemetry: null,
    trackRoles: [],
  });

  useEffect(() => {
    const peer = peerRef.current;
    if (peer !== null) {
      peer.close();
      peerRef.current = null;
    }
    if (rgbVideoRef.current !== null) {
      rgbVideoRef.current.srcObject = null;
    }
    if (depthVideoRef.current !== null) {
      depthVideoRef.current.srcObject = null;
    }
    if (!enabled) {
      setState({
        connected: false,
        error: "",
        session: null,
        snapshot: null,
        telemetry: null,
        trackRoles: [],
      });
      return;
    }

    let cancelled = false;
    let trackIndex = 0;

    async function connect() {
      try {
        const config = await requestJson<{
          iceServers: Array<{ urls: string[] }>;
          enableDepthTrack: boolean;
        }>(`${basePath}/config`);
        if (cancelled) {
          return;
        }
        const connection = new RTCPeerConnection({
          iceServers: config.iceServers ?? [],
        });
        peerRef.current = connection;

        const stateChannel = connection.createDataChannel("state");
        const telemetryChannel = connection.createDataChannel("telemetry", {
          ordered: false,
          maxRetransmits: 0,
        });
        const trackCount = config.enableDepthTrack ? 2 : 1;
        for (let index = 0; index < trackCount; index += 1) {
          connection.addTransceiver("video", { direction: "recvonly" });
        }

        connection.ontrack = (event) => {
          const stream = new MediaStream([event.track]);
          if (trackIndex === 0 && rgbVideoRef.current !== null) {
            rgbVideoRef.current.srcObject = stream;
          } else if (trackIndex === 1 && depthVideoRef.current !== null) {
            depthVideoRef.current.srcObject = stream;
          }
          trackIndex += 1;
        };
        connection.onconnectionstatechange = () => {
          setState((current) => ({
            ...current,
            connected: connection.connectionState === "connected",
            error: connection.connectionState === "failed" ? "WebRTC 연결이 실패했습니다." : current.error,
          }));
        };
        stateChannel.onmessage = (event) => {
          const payload = JSON.parse(String(event.data)) as ViewerStateMessage;
          const type = String(payload.type ?? "");
          setState((current) => {
            if (type === "session_ready") {
              const trackRoles = Array.isArray(payload.trackRoles)
                ? payload.trackRoles.map((item) => String(item))
                : current.trackRoles;
              return { ...current, session: payload, trackRoles };
            }
            if (type === "snapshot" || type === "waiting_for_frame") {
              return { ...current, snapshot: payload };
            }
            return current;
          });
        };
        telemetryChannel.onmessage = (event) => {
          const payload = JSON.parse(String(event.data)) as ViewerTelemetryMessage;
          if (String(payload.type ?? "") !== "frame_meta") {
            return;
          }
          setState((current) => ({ ...current, telemetry: payload }));
        };

        const offer = await connection.createOffer();
        await connection.setLocalDescription(offer);
        await waitForIceGatheringComplete(connection);
        const answer = await requestJson<{ sdp: string; type: RTCSdpType }>(`${basePath}/offer`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sdp: connection.localDescription?.sdp ?? "",
            type: connection.localDescription?.type ?? "offer",
          }),
        });
        if (cancelled) {
          return;
        }
        await connection.setRemoteDescription(answer);
      } catch (error) {
        if (!cancelled) {
          setState((current) => ({
            ...current,
            error: error instanceof Error ? error.message : String(error),
          }));
        }
      }
    }

    void connect();
    return () => {
      cancelled = true;
      if (peerRef.current !== null) {
        peerRef.current.close();
        peerRef.current = null;
      }
    };
  }, [basePath, enabled]);

  return { rgbVideoRef, depthVideoRef, ...state };
}
