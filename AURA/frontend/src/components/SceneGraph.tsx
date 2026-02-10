import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, Search } from "lucide-react";
import { useStreaming } from "../application/streaming/StreamingProvider";
import { EMPTY_METADATA } from "../domain/streaming/metadataParsers";
import type { StreamRelation } from "../domain/streaming/streamTypes";
import { ForcedGraph, type GraphData, type LinkData, type NodeData } from "./graph/ForcedGraph";

const STATIC_COLOR = "#4fd1c5";
const TEMPORAL_COLOR = "#805ad5";

const FALLBACK_TRIPLETS = [
  { subject: "person", subject_id: 3, predicate: "sitting on", object: "chair", object_id: 7, confidence: 0.92, type: "static" as const },
  { subject: "cup", subject_id: 18, predicate: "on", object: "table", object_id: 2, confidence: 0.88, type: "static" as const },
  { subject: "person", subject_id: 3, predicate: "looking at", object: "monitor", object_id: 15, confidence: 0.79, type: "temporal" as const },
  { subject: "laptop", subject_id: 4, predicate: "on", object: "table", object_id: 2, confidence: 0.95, type: "static" as const },
  { subject: "chair", subject_id: 7, predicate: "next to", object: "table", object_id: 2, confidence: 0.85, type: "static" as const },
  { subject: "monitor", subject_id: 15, predicate: "on", object: "desk", object_id: 9, confidence: 0.91, type: "static" as const },
  { subject: "person", subject_id: 3, predicate: "holding", object: "cup", object_id: 18, confidence: 0.73, type: "temporal" as const },
];

type Triplet = {
  subject: string;
  subject_id: number;
  predicate: string;
  object: string;
  object_id: number;
  confidence: number;
  type: "static" | "temporal";
};

export function SceneGraph() {
  const { metadata: streamingMetadata } = useStreaming();
  const metadata = streamingMetadata ?? EMPTY_METADATA;

  const lastSignatureRef = useRef<string | null>(null);
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
  const [triplets, setTriplets] = useState<Triplet[]>(FALLBACK_TRIPLETS);

  const buildSignature = useMemo(() => {
    const relations = Array.isArray(metadata.relations) ? metadata.relations : [];
    const relPart = relations
      .map((r) => ({
        s: r.subjectEntityId ?? null,
        o: r.objectEntityId ?? null,
        rel: r.relation ?? "",
        t: r.type ?? "",
        c: r.confidence ?? null,
      }))
      .sort((a, b) => (a.s ?? 0) - (b.s ?? 0) || (a.o ?? 0) - (b.o ?? 0) || a.rel.localeCompare(b.rel));
    const recordPart = (metadata.entityRecords ?? [])
      .map((r) => ({ id: r.entityId, cls: r.baseCls }))
      .sort((a, b) => a.id - b.id);
    const entityPart = (metadata.entities ?? [])
      .map((e) => ({ id: e.entityId, cls: e.cls }))
      .sort((a, b) => a.id - b.id);
    return JSON.stringify({ relPart, recordPart, entityPart });
  }, [metadata.relations, metadata.entityRecords, metadata.entities]);

  useEffect(() => {
    if (lastSignatureRef.current === buildSignature) return;
    lastSignatureRef.current = buildSignature;

    const relations: StreamRelation[] = Array.isArray(metadata.relations) ? metadata.relations : [];
    const nodesMap = new Map<string, NodeData>();
    const addNode = (id: string, name: string) => {
      if (!nodesMap.has(id)) {
        nodesMap.set(id, { id, name, group: name === "person" ? 1 : 2 });
      }
    };

    metadata.entityRecords.forEach((rec) => addNode(`${rec.entityId}`, rec.baseCls || "entity"));
    metadata.entities.forEach((ent) => addNode(`${ent.entityId}`, ent.cls || "entity"));

    const relationTriplets: Triplet[] = relations.length
      ? relations.map((rel, idx) => {
          const subjectId = rel.subjectEntityId ?? idx;
          const objectId = rel.objectEntityId ?? idx + 1000;
          const subjectName =
            rel.subjectCls ||
            metadata.entityRecords.find((r) => r.entityId === subjectId)?.baseCls ||
            metadata.entities.find((e) => e.entityId === subjectId)?.cls ||
            "entity";
          const objectName =
            rel.objectCls ||
            metadata.entityRecords.find((r) => r.entityId === objectId)?.baseCls ||
            metadata.entities.find((e) => e.entityId === objectId)?.cls ||
            "entity";
          addNode(`${subjectId}`, subjectName);
          addNode(`${objectId}`, objectName);
          return {
            subject: subjectName,
            subject_id: subjectId,
            predicate: rel.relation || "related to",
            object: objectName,
            object_id: objectId,
            confidence: typeof rel.confidence === "number" ? rel.confidence : 0.75,
            type: rel.type === "temporal" ? "temporal" : "static",
          } satisfies Triplet;
        })
      : FALLBACK_TRIPLETS;

    relationTriplets.forEach((t) => {
      addNode(`${t.subject_id}`, t.subject);
      addNode(`${t.object_id}`, t.object);
    });

    const links: LinkData[] = relationTriplets.map((t): LinkData => ({
      source: `${t.subject_id}`,
      target: `${t.object_id}`,
      predicate: t.predicate,
      type: t.type,
      confidence: t.confidence,
    }));

    const graph: GraphData = { nodes: Array.from(nodesMap.values()), links };
    setGraphData(graph);
    setTriplets(relationTriplets);
  }, [buildSignature, metadata]);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 500 });

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

  return (
    <div className="flex h-full flex-col space-y-6">
      {/* Subtitle */}
      <p className="text-[14px] text-[#a0aec0]">
        Current scene graph and relational triplets
      </p>

      {/* Two Column Layout */}
      <div className="grid h-full min-h-0 grid-cols-[1fr_450px] gap-6">
        {/* LEFT COLUMN - Scene Graph Canvas */}
        <div className="flex h-full flex-col rounded-[15px] bg-white p-6 shadow-[0px_3.5px_5.5px_0px_rgba(0,0,0,0.02)]">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-[18px] text-[#2d3748]">Scene Graph</h3>

            <div className="flex items-center gap-2">
              {/* Layout Dropdown */}
              <button className="flex items-center gap-2 rounded-lg border border-[#e2e8f0] bg-white px-3 py-2 text-[12px] text-[#2d3748] hover:bg-gray-50">
                <span>Force-directed</span>
                <ChevronDown className="size-3" />
              </button>
            </div>
          </div>

          {/* Filter chips */}
          <div className="mb-4 flex flex-wrap gap-2">
            <button className="rounded-lg bg-[#4fd1c5] px-3 py-1 text-[12px] text-white">
              All
            </button>
            <button className="rounded-lg bg-gray-100 px-3 py-1 text-[12px] text-[#a0aec0] hover:bg-gray-200">
              Static
            </button>
            <button className="rounded-lg bg-gray-100 px-3 py-1 text-[12px] text-[#a0aec0] hover:bg-gray-200">
              Temporal
            </button>
          </div>

          {/* Graph Canvas */}
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
            />
          </div>

          {/* Legend */}
          <div className="mt-4 flex items-center justify-center gap-6 text-[12px] text-[#a0aec0]">
            <div className="flex items-center gap-2">
              <div className="h-0.5 w-6 rounded" style={{ backgroundColor: STATIC_COLOR }}></div>
              <span>Static relation</span>
            </div>
            <div className="flex items-center gap-2">
              <svg width="24" height="2">
                <line x1="0" y1="1" x2="24" y2="1" stroke={TEMPORAL_COLOR} strokeWidth="2" strokeDasharray="4,4" />
              </svg>
              <span>Temporal relation</span>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN - Triplets */}
        <div className="flex min-h-0 flex-col">
          {/* Relation Triplets Table */}
          <div className="flex h-full flex-col rounded-[15px] bg-white p-6 shadow-[0px_3.5px_5.5px_0px_rgba(0,0,0,0.02)]">
            <h3 className="mb-4 text-[18px] text-[#2d3748]">Relation Triplets</h3>

            {/* Search/Filter */}
            <div className="mb-4 flex items-center gap-2">
              <div className="flex flex-1 items-center gap-2 rounded-lg border border-[#e2e8f0] bg-white px-3 py-2">
                <Search className="size-4 text-[#a0aec0]" />
                <input
                  type="text"
                  placeholder="Filter by label or relation"
                  className="flex-1 border-0 bg-transparent text-[12px] text-[#2d3748] outline-none"
                />
              </div>
            </div>

            {/* Filter chips */}
            <div className="mb-4 flex flex-wrap gap-2">
              <button className="rounded-lg bg-[#4fd1c5] px-2 py-1 text-[11px] text-white">
                All
              </button>
              <button className="rounded-lg bg-gray-100 px-2 py-1 text-[11px] text-[#a0aec0] hover:bg-gray-200">
                People
              </button>
              <button className="rounded-lg bg-gray-100 px-2 py-1 text-[11px] text-[#a0aec0] hover:bg-gray-200">
                Objects
              </button>
              <button className="rounded-lg bg-gray-100 px-2 py-1 text-[11px] text-[#a0aec0] hover:bg-gray-200">
                Spatial
              </button>
              <button className="rounded-lg bg-gray-100 px-2 py-1 text-[11px] text-[#a0aec0] hover:bg-gray-200">
                Interaction
              </button>
            </div>

            {/* Table */}
            <div className="flex-1 overflow-y-auto">
              {/* Table Header */}
              <div className="sticky top-0 mb-3 grid grid-cols-[1fr_1.2fr_1fr_100px] gap-3 border-b border-[#e2e8f0] bg-white pb-2">
                <div className="text-[10px] uppercase tracking-wide text-[#a0aec0]">Subject</div>
                <div className="text-[10px] uppercase tracking-wide text-[#a0aec0]">Predicate</div>
                <div className="text-[10px] uppercase tracking-wide text-[#a0aec0]">Object</div>
                <div className="text-[10px] uppercase tracking-wide text-[#a0aec0]">Conf.</div>
              </div>

              {/* Table Rows */}
              <div className="space-y-3">
                {triplets.map((triplet, index) => (
                  <div
                    key={`${triplet.subject_id}-${triplet.object_id}-${triplet.predicate}-${index}`}
                    className="grid grid-cols-[1fr_1.2fr_1fr_100px] items-center gap-3 rounded-lg border border-[#e2e8f0] p-3 hover:bg-gray-50"
                  >
                    {/* Subject */}
                    <div>
                      <div className="text-[13px] text-[#2d3748] capitalize">{triplet.subject}</div>
                      <div className="text-[10px] text-[#a0aec0]">#{triplet.subject_id}</div>
                    </div>

                    {/* Predicate with Type Tag */}
                    <div>
                      <div className="mb-1 text-[13px] text-[#a0aec0]">{triplet.predicate}</div>
                      <span
                        className={`inline-block rounded-md px-2 py-0.5 text-[10px] ${
                          triplet.type === "static"
                            ? "bg-[#4fd1c5]/20 text-[#4fd1c5]"
                            : "bg-[#805ad5]/20 text-[#805ad5]"
                        }`}
                      >
                        {triplet.type}
                      </span>
                    </div>

                    {/* Object */}
                    <div>
                      <div className="text-[13px] text-[#2d3748] capitalize">{triplet.object}</div>
                      <div className="text-[10px] text-[#a0aec0]">#{triplet.object_id}</div>
                    </div>

                    {/* Confidence */}
                    <div>
                      <div className="mb-1 text-right text-[12px] text-[#4fd1c5]">
                        {Math.round(triplet.confidence * 100)}%
                      </div>
                      <div className="h-1 w-full rounded-full bg-[#e2e8f0]">
                        <div
                          className="h-1 rounded-full bg-[#4fd1c5]"
                          style={{ width: `${triplet.confidence * 100}%` }}
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
