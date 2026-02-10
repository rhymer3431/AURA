import { useEffect, useMemo, useRef, useState } from "react";
import { useStreaming } from "../application/streaming/StreamingProvider";
import type { GraphData, NodeData } from "../domain/graph/ForcedGraphProps";
import { InternalGraphState } from "../domain/graph/InternalGraphState";
import { Triplet } from "../domain/graph/Triplet";
import { applySceneGraphDiff } from "../infra/graph/applyGraphDiff";
import { extractSceneGraphDiff } from "../infra/graph/extractSceneGraphDiff";
import { hasDiffContent } from "../infra/graph/hasDiffContent";
import { EMPTY_METADATA } from "../infra/streaming/metadataParsers";
import { ForcedGraph } from "./graph/ForcedGraph";

const STATIC_COLOR = "#4fd1c5";
const FALLBACK_TRIPLETS: Triplet[] = [];

interface NodePosition {
  x: number;
  y: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
}

export function SceneGraph() {
  const { metadata: streamingMetadata } = useStreaming();
  const metadata = streamingMetadata ?? EMPTY_METADATA;

  const metadataRef = useRef(metadata);
  useEffect(() => {
    metadataRef.current = metadata;
  }, [metadata]);

  const sceneGraphDiff = useMemo(
    () => extractSceneGraphDiff(metadata),
    [metadata],
  );

  const graphStateRef = useRef<InternalGraphState>({
    nodes: new Map(),
    links: new Map(),
    triplets: [...FALLBACK_TRIPLETS],
  });

  const positionsRef = useRef<Map<string, NodePosition>>(new Map());

  const [graphVersion, setGraphVersion] = useState(0);
  const [triplets, setTriplets] = useState<Triplet[]>(FALLBACK_TRIPLETS);

  useEffect(() => {
    if (!sceneGraphDiff || !hasDiffContent(sceneGraphDiff)) return;

    const safeMeta = metadataRef.current ?? EMPTY_METADATA;
    const updated = applySceneGraphDiff(
      graphStateRef.current,
      sceneGraphDiff,
      safeMeta,
      positionsRef.current,
    );

    graphStateRef.current = updated;
    setTriplets(updated.triplets);
    setGraphVersion((v) => v + 1);
  }, [sceneGraphDiff]);

  const graphData: GraphData = useMemo(() => {
    const baseNodes = Array.from(graphStateRef.current.nodes.values());
    const baseLinks = Array.from(graphStateRef.current.links.values());

    if (baseNodes.length === 0) {
      return {
        nodes: [
          { id: "sample-1", name: "Sample 1", group: 1, x: 300, y: 250 },
          { id: "sample-2", name: "Sample 2", group: 2, x: 500, y: 250 },
        ],
        links: [{ source: "sample-1", target: "sample-2", predicate: "related", type: "static" }],
      };
    }

    const nodesWithPositions = baseNodes.map((node) => {
      const hint = positionsRef.current.get(node.id);
      return hint ? { ...node, ...hint } : node;
    });

    return {
      nodes: nodesWithPositions,
      links: baseLinks,
    };
  }, [graphVersion]);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const [dimensions, setDimensions] = useState({
    width: 800,
    height: 500,
  });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const updateSize = () => {
      setDimensions({
        width: el.clientWidth,
        height: Math.max(360, el.clientHeight - 32),
      });
    };

    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const colorFn = (node: NodeData) => {
    if (node.group === 1) return STATIC_COLOR;
    if (node.group === 2) return "#4299e1";
    return STATIC_COLOR;
  };

  const handleNodePositionUpdate = (nodeId: string, x: number, y: number, vx?: number, vy?: number) => {
    positionsRef.current.set(nodeId, { x, y, vx, vy });
  };

  return (
    <div className="flex h-full flex-col space-y-6">
      <p className="text-[14px] text-[#a0aec0]">
        Current scene graph and relational triplets
      </p>

      <div className="grid h-full min-h-0 grid-cols-[1fr_450px] gap-6">
        <div className="flex h-full flex-col rounded-[15px] bg-white p-6 shadow-[0px_3.5px_5.5px_0px_rgba(0,0,0,0.02)]">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-[18px] text-[#2d3748]">
              Scene Graph
            </h3>
            <div className="text-[12px] text-[#a0aec0]">
              {graphData.nodes.length} nodes, {graphData.links.length} edges
            </div>
          </div>

          <div
            ref={containerRef}
            className="relative flex-1 overflow-hidden rounded-lg bg-gradient-to-br from-gray-50 to-gray-100"
          >
            <ForcedGraph
              data={graphData}
              width={dimensions.width}
              height={dimensions.height}
              nodeSize={12}
              colorFunction={colorFn}
              onNodePositionUpdate={handleNodePositionUpdate}
            />
          </div>

          <div className="mt-4 flex items-center justify-center gap-6 text-[12px] text-[#a0aec0]">
            <div className="flex items-center gap-2">
              <div
                className="h-0.5 w-6 rounded"
                style={{ backgroundColor: STATIC_COLOR }}
              ></div>
              <span>Static relation</span>
            </div>
            <div className="flex items-center gap-2">
              <svg width="24" height="2">
                <line
                  x1="0"
                  y1="1"
                  x2="24"
                  y2="1"
                  stroke="#805ad5"
                  strokeWidth="2"
                  strokeDasharray="4,4"
                />
              </svg>
              <span>Temporal relation</span>
            </div>
          </div>
        </div>

        <div className="flex min-h-0 flex-col rounded-[15px] bg-white p-6 shadow-[0px_3.5px_5.5px_0px_rgba(0,0,0,0.02)]">
          <h3 className="mb-4 text-[18px] text-[#2d3748]">
            Relation Triplets ({triplets.length})
          </h3>

          <div className="flex-1 overflow-auto">
            <div className="min-w-[560px]">
              <div className="sticky top-0 mb-3 grid grid-cols-[1fr_1.2fr_1fr_100px] gap-3 border-b border-[#e2e8f0] bg-white pb-2">
                <div className="text-[10px] uppercase tracking-wide text-[#a0aec0]">
                  Subject
                </div>
                <div className="text-[10px] uppercase tracking-wide text-[#a0aec0]">
                  Predicate
                </div>
                <div className="text-[10px] uppercase tracking-wide text-[#a0aec0]">
                  Object
                </div>
                <div className="text-[10px] uppercase tracking-wide text-[#a0aec0]">
                  Conf.
                </div>
              </div>

              <div className="space-y-3">
                {triplets.map((triplet, index) => (
                  <div
                    key={`${triplet.subject_id}-${triplet.object_id}-${triplet.predicate}-${index}`}
                    className="grid grid-cols-[1fr_1.2fr_1fr_100px] items-center gap-3 rounded-lg border border-[#e2e8f0] p-3 transition-colors hover:bg-gray-50"
                  >
                    <div>
                      <div className="text-[13px] capitalize text-[#2d3748]">
                        {triplet.subject}
                      </div>
                      <div className="text-[10px] text-[#a0aec0]">
                        #{triplet.subject_id}
                      </div>
                    </div>

                    <div>
                      <div className="mb-1 text-[13px] text-[#a0aec0]">
                        {triplet.predicate}
                      </div>
                      <span
                        className={`inline-block rounded-md px-2 py-0.5 text-[10px] ${triplet.type === "static"
                          ? "bg-[#4fd1c5]/20 text-[#4fd1c5]"
                          : "bg-[#805ad5]/20 text-[#805ad5]"
                          }`}
                      >
                        {triplet.type}
                      </span>
                    </div>

                    <div>
                      <div className="text-[13px] capitalize text-[#2d3748]">
                        {triplet.object}
                      </div>
                      <div className="text-[10px] text-[#a0aec0]">
                        #{triplet.object_id}
                      </div>
                    </div>

                    <div>
                      <div className="mb-1 text-right text-[12px] text-[#4fd1c5]">
                        {Math.round(triplet.confidence * 100)}%
                      </div>
                      <div className="h-1 w-full rounded-full bg-[#e2e8f0]">
                        <div
                          className="h-1 rounded-full bg-[#4fd1c5] transition-all duration-300"
                          style={{
                            width: `${triplet.confidence * 100}%`,
                          }}
                        ></div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
