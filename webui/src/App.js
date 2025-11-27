import { useEffect, useRef, useState } from "react";
import CytoscapeComponent from "react-cytoscapejs";

const WS_URL = "ws://localhost:7000/ws";
const VIDEO_URL = "http://localhost:7000/video";

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

function App() {
  const cyRef = useRef(null);
  const [elements, setElements] = useState([]);

  const [retryToken, setRetryToken] = useState(Date.now());
  const videoSrc = `${VIDEO_URL}?t=${retryToken}`;

  useEffect(() => {
    const interval = setInterval(() => {
      setRetryToken(Date.now());
    }, 10000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    let ws;
    let retryTimer;

    const connect = () => {
      ws = new WebSocket(WS_URL);

      ws.onmessage = (evt) => {
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

      ws.onopen = () => console.log("WS connected");
      ws.onclose = () => {
        console.log("WS closed, retrying...");
        retryTimer = setTimeout(connect, 2000);
      };
      ws.onerror = (e) => console.error("WS error", e);
    };

    connect();

    return () => {
      if (ws) ws.close();
      if (retryTimer) clearTimeout(retryTimer);
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
        <img
          src={videoSrc}
          alt="video stream"
          onError={() => setRetryToken(Date.now())}
          style={{ width: "100%", height: "100%", objectFit: "cover" }}
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
