import { useEffect, useRef, useState } from "react";
import CytoscapeComponent from "react-cytoscapejs";

const OFFER_URL = "http://localhost:7000/offer";

const layout = { name: "cose", animate: true };
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

  useEffect(() => {
    let cancelled = false;

    async function startWebRTC() {
      const pc = new RTCPeerConnection({ iceServers: [] });
      pcRef.current = pc;

      const graphChannel = pc.createDataChannel("graph");
      graphChannel.onmessage = (evt) => {
        try {
          const graph = JSON.parse(evt.data);
          const newElements = toElements(graph);
          setElements(newElements);
          if (cyRef.current) {
            cyRef.current.json({ elements: newElements });
            cyRef.current.layout(layout).run();
          }
        } catch (err) {
          console.error("Failed to parse graph", err);
        }
      };

      pc.ontrack = (event) => {
        const [stream] = event.streams;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      };

      await pc.setLocalDescription(await pc.createOffer({ offerToReceiveVideo: true }));
      await waitForIceGathering(pc);

      if (cancelled) return;

      const offer = pc.localDescription;
      const res = await fetch(OFFER_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sdp: offer.sdp, type: offer.type }),
      });
      const answer = await res.json();
      await pc.setRemoteDescription(answer);
    }

    startWebRTC();

    return () => {
      cancelled = true;
      if (pcRef.current) {
        pcRef.current.close();
      }
    };
  }, []);

  return (
    <div
      style={{
        background: "#0f0f0f",
        width: "100vw",
        height: "100vh",
        margin: 0,
        padding: 0,
        overflow: "hidden",
        display: "flex",
      }}
    >
      <div style={{ flex: 1, borderRight: "1px solid #1f1f1f" }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{ width: "100%", height: "100%", objectFit: "cover", background: "#000" }}
        />
      </div>

      <div style={{ flex: 1 }}>
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
    </div>
  );
}

export default App;
