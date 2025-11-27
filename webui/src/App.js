import { useEffect, useRef, useState } from "react";
import CytoscapeComponent from "react-cytoscapejs";

const OFFER_URL = "http://localhost:7000/offer";

const layout = { name: "cose", animate: false, nodeRepulsion: 8000, refresh: 20 };
const style = [
  {
    selector: "node",
    style: {
      "background-color": "#39ff14",
      "border-color": "#0affff",
      "border-width": 2,
      label: "data(label)",
      color: "#e0f7ff",
      "font-size": 12,
      "text-outline-color": "#0f0f0f",
      "text-outline-width": 2,
    },
  },
  {
    selector: "edge",
    style: {
      width: 2,
      "line-color": "#3b7bff",
      "target-arrow-color": "#3b7bff",
      "target-arrow-shape": "triangle",
      label: "data(label)",
      color: "#9dc4ff",
      "font-size": 11,
      "text-background-opacity": 0.6,
      "text-background-color": "#0f0f0f",
      "curve-style": "bezier",
    },
  },
];

function toElements(graph) {
  const nodes = (graph.nodes || []).map((n) => ({
    data: { id: String(n.id), label: n.label ?? n.id },
  }));
  const edges = (graph.edges || []).map((e, idx) => ({
    data: {
      id: `e-${idx}-${e.source}-${e.target}`,
      source: String(e.source),
      target: String(e.target),
      label: e.label ?? "",
    },
  }));
  return [...nodes, ...edges];
}

function waitForIceGathering(pc) {
  return new Promise((resolve) => {
    if (pc.iceGatheringState === "complete") {
      resolve();
    } else {
      const checkState = () => {
        if (pc.iceGatheringState === "complete") {
          pc.removeEventListener("icegatheringstatechange", checkState);
          resolve();
        }
      };
      pc.addEventListener("icegatheringstatechange", checkState);
    }
  });
}

function App() {
  const cyRef = useRef(null);
  const videoRef = useRef(null);
  const pcRef = useRef(null);
  const [elements, setElements] = useState([]);
  const [loading, setLoading] = useState(true);
  const [statusMessage, setStatusMessage] = useState("Connecting stream...");
  const lastGraphJson = useRef("");
  const layoutQueued = useRef(false);
  const retryTimer = useRef(null);

  useEffect(() => {
    let cancelled = false;

    async function startWebRTC() {
      // cleanup any existing pc
      if (pcRef.current) {
        try {
          pcRef.current.close();
        } catch (_) {}
      }
      const pc = new RTCPeerConnection({
        iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
        bundlePolicy: "max-bundle",
        rtcpMuxPolicy: "require",
      });
      pcRef.current = pc;

      const graphChannel = pc.createDataChannel("graph", {
        ordered: false,
        maxRetransmits: 0, // drop old updates to avoid head-of-line blocking
      });
      graphChannel.onmessage = (evt) => {
        try {
          if (evt.data === lastGraphJson.current) return;
          lastGraphJson.current = evt.data;
          const graph = JSON.parse(evt.data);
          const newElements = toElements(graph);
          setElements(newElements);
          if (cyRef.current && !layoutQueued.current) {
            layoutQueued.current = true;
            requestAnimationFrame(() => {
              cyRef.current.json({ elements: newElements });
              cyRef.current.layout(layout).run();
              layoutQueued.current = false;
            });
          }
        } catch (err) {
          console.error("Failed to parse graph", err);
        }
      };

      pc.ontrack = (event) => {
        const [stream] = event.streams;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setLoading(false);
          setStatusMessage("");
        }
      };

      pc.oniceconnectionstatechange = () => {
        const state = pc.iceConnectionState;
        if (state === "failed" || state === "disconnected") {
          setLoading(true);
          setStatusMessage("Connection lost. Retrying...");
          if (retryTimer.current) clearTimeout(retryTimer.current);
          retryTimer.current = setTimeout(() => startWebRTC(), 1500);
        }
      };

      try {
        setLoading(true);
        setStatusMessage("Connecting stream...");
        await pc.setLocalDescription(await pc.createOffer({ offerToReceiveVideo: true }));
        await waitForIceGathering(pc);

        if (cancelled) return;

        const offer = pc.localDescription;
        const res = await fetch(OFFER_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sdp: offer.sdp, type: offer.type }),
        });
        if (!res.ok) {
          throw new Error(`Signaling failed: ${res.status}`);
        }
        const answer = await res.json();
        await pc.setRemoteDescription(answer);
      } catch (err) {
        console.error("WebRTC connection failed", err);
        if (cancelled) return;
        setLoading(true);
        setStatusMessage("Connection failed. Retrying...");
        if (retryTimer.current) clearTimeout(retryTimer.current);
        retryTimer.current = setTimeout(() => {
          startWebRTC();
        }, 2000);
      }
    }

    startWebRTC();

    return () => {
      cancelled = true;
      if (retryTimer.current) {
        clearTimeout(retryTimer.current);
      }
      if (pcRef.current) {
        pcRef.current.close();
      }
    };
  }, []);

  return (
    <div style={styles.shell}>
      {loading && (
        <div style={styles.loadingOverlay}>
          <div style={styles.spinner} />
          <div style={styles.loadingText}>{statusMessage}</div>
        </div>
      )}
      <header style={styles.header}>
        <div style={styles.logoDot} />
        <div>
          <div style={styles.title}>Scene Graph Monitor</div>
          <div style={styles.subtitle}>Live WebRTC video + relations</div>
        </div>
      </header>

      <div style={styles.grid}>
        <section style={styles.videoCard}>
          <div style={styles.cardHeader}>
            <span>Live Video</span>
            <span style={styles.badge}>WebRTC</span>
          </div>
          <div style={styles.cardBody}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={styles.video}
            />
          </div>
        </section>

        <section style={styles.graphCard}>
          <div style={styles.cardHeader}>
            <span>Scene Graph</span>
            <span style={styles.badgeSecondary}>Relations</span>
          </div>
          <div style={styles.cardBody}>
            <CytoscapeComponent
              cy={(cy) => {
                cyRef.current = cy;
              }}
              elements={elements}
              layout={layout}
              stylesheet={style}
              style={{ width: "100%", height: "100%" }}
            />
          </div>
        </section>
      </div>
    </div>
  );
}

export default App;

const styles = {
  shell: {
    background: "radial-gradient(circle at 20% 20%, #1a1f2b 0, #0b0c10 35%)",
    width: "100vw",
    height: "100vh",
    margin: 0,
    padding: "16px",
    boxSizing: "border-box",
    color: "#e0f7ff",
    display: "flex",
    flexDirection: "column",
    gap: "12px",
  },
  header: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    padding: "10px 12px",
    borderRadius: "10px",
    background: "rgba(255, 255, 255, 0.04)",
    border: "1px solid rgba(255, 255, 255, 0.06)",
    boxShadow: "0 10px 30px rgba(0,0,0,0.35)",
  },
  logoDot: {
    width: "14px",
    height: "14px",
    borderRadius: "50%",
    background: "linear-gradient(135deg, #39ff14, #0affff)",
    boxShadow: "0 0 12px #39ff14",
  },
  title: {
    fontSize: "16px",
    fontWeight: 700,
    letterSpacing: "0.4px",
  },
  subtitle: {
    fontSize: "12px",
    color: "#9db5c9",
  },
  grid: {
    flex: 1,
    display: "grid",
    gridTemplateColumns: "2fr 1fr",
    gap: "12px",
    minHeight: 0,
  },
  videoCard: {
    background: "rgba(12, 16, 24, 0.9)",
    borderRadius: "14px",
    border: "1px solid rgba(58, 94, 130, 0.4)",
    boxShadow: "0 20px 60px rgba(0,0,0,0.45)",
    display: "flex",
    flexDirection: "column",
    minHeight: 0,
  },
  graphCard: {
    background: "rgba(12, 16, 24, 0.9)",
    borderRadius: "14px",
    border: "1px solid rgba(58, 94, 130, 0.4)",
    boxShadow: "0 20px 60px rgba(0,0,0,0.45)",
    display: "flex",
    flexDirection: "column",
    minHeight: 0,
  },
  cardHeader: {
    padding: "12px 14px",
    borderBottom: "1px solid rgba(255,255,255,0.05)",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    fontSize: "13px",
    fontWeight: 600,
    letterSpacing: "0.3px",
  },
  badge: {
    padding: "4px 8px",
    borderRadius: "12px",
    background: "linear-gradient(135deg, #39ff14, #0affff)",
    color: "#041018",
    fontSize: "11px",
    fontWeight: 700,
  },
  badgeSecondary: {
    padding: "4px 8px",
    borderRadius: "12px",
    background: "rgba(59,123,255,0.2)",
    color: "#9dc4ff",
    fontSize: "11px",
    fontWeight: 700,
    border: "1px solid rgba(59,123,255,0.5)",
  },
  cardBody: {
    flex: 1,
    minHeight: 0,
  },
  video: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
    borderRadius: "0 0 14px 14px",
    background: "#000",
  },
  loadingOverlay: {
    position: "fixed",
    top: 0,
    left: 0,
    width: "100vw",
    height: "100vh",
    background: "rgba(0,0,0,0.75)",
    backdropFilter: "blur(4px)",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 999,
    color: "#e0f7ff",
    gap: "12px",
  },
  spinner: {
    width: "56px",
    height: "56px",
    border: "4px solid rgba(255,255,255,0.15)",
    borderTop: "4px solid #39ff14",
    borderRadius: "50%",
    animation: "spin 1s linear infinite",
  },
  loadingText: {
    fontSize: "14px",
    letterSpacing: "0.3px",
    color: "#b8d8ff",
  },
};
