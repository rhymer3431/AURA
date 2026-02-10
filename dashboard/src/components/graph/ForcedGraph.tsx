/* eslint-disable react-hooks/exhaustive-deps */
import * as d3 from "d3";
import React, { useCallback, useEffect, useRef } from "react";

import type {
    GraphData,
    NodeData,
} from "../../domain/graph/ForcedGraphProps";
import { LinkData } from "../../domain/graph/LinkData";

interface ForcedGraphProps {
    data: GraphData;
    width?: number;
    height?: number;
    nodeSize?: number;
    onNodeClick?: (node: NodeData) => void;
    colorFunction?: (node: NodeData) => string;
    onNodePositionUpdate?: (nodeId: string, x: number, y: number, vx?: number, vy?: number) => void;
}

const STATIC_COLOR = "#4fd1c5";
const TEMPORAL_COLOR = "#805ad5";

export const ForcedGraph: React.FC<ForcedGraphProps> = ({
    data,
    width = 800,
    height = 500,
    nodeSize = 12,
    onNodeClick,
    colorFunction = () => "#4E91F9",
    onNodePositionUpdate,
}) => {
    const svgRef = useRef<SVGSVGElement | null>(null);
    const simulationRef = useRef<d3.Simulation<NodeData, LinkData> | null>(null);
    const prevNodeIdsRef = useRef<Set<string>>(new Set());

    // 🔥 D3 selection들을 저장하여 재사용
    const selectionsRef = useRef<{
        zoomLayer?: d3.Selection<SVGGElement, unknown, null, undefined>;
        linkGroup?: d3.Selection<SVGGElement, unknown, null, undefined>;
        nodeLabelGroup?: d3.Selection<SVGGElement, unknown, null, undefined>;
        linkLabelGroup?: d3.Selection<SVGGElement, unknown, null, undefined>;
    }>({});

    // 🔥 초기 렌더링 여부 추적
    const isInitialRenderRef = useRef(true);
    const prevDataRef = useRef<GraphData>({ nodes: [], links: [] });

    const drawGraph = useCallback(() => {
        const svgEl = svgRef.current;
        if (!svgEl) return;

        // 🔥 데이터가 실제로 변경되었는지 확인
        const dataChanged =
            prevDataRef.current.nodes.length !== data.nodes.length ||
            prevDataRef.current.links.length !== data.links.length;

        if (!dataChanged && !isInitialRenderRef.current) {
            console.log('⏭️ Skipping drawGraph - no data change');
            return;
        }

        console.log('🎨 ForcedGraph drawGraph called', {
            nodes: data.nodes.length,
            links: data.links.length,
            width,
            height,
            isInitial: isInitialRenderRef.current
        });

        prevDataRef.current = { nodes: [...data.nodes], links: [...data.links] };

        const svg = d3.select(svgEl);

        // 🔥 초기 렌더링시에만 기본 구조 생성
        if (isInitialRenderRef.current) {
            svg.selectAll("*").remove();
            const zoomLayer = svg.append("g");
            const linkGroup = zoomLayer.append("g").attr("class", "links");
            const linkLabelGroup = zoomLayer.append("g").attr("class", "link-labels");
            const nodeGroup = zoomLayer.append("g").attr("class", "nodes");
            const nodeLabelGroup = zoomLayer.append("g").attr("class", "node-labels");

            selectionsRef.current = {
                zoomLayer,
                linkGroup,
                nodeLabelGroup: nodeGroup,
                linkLabelGroup,
            };

            // Zoom behavior 설정 (한 번만)
            const zoomBehavior = d3
                .zoom<SVGSVGElement, unknown>()
                .scaleExtent([0.1, 4])
                .on("zoom", (event) => {
                    zoomLayer.attr("transform", event.transform);
                });

            svg.call(zoomBehavior as any);

            isInitialRenderRef.current = false;
        }

        const { linkGroup, nodeLabelGroup, linkLabelGroup } = selectionsRef.current;
        if (!linkGroup || !nodeLabelGroup || !linkLabelGroup) return;

        // --- 새로 추가된 노드 식별 ---
        const prevIds = prevNodeIdsRef.current;
        const currentIds = new Set<string>(data.nodes.map((n) => n.id));
        const newIds = new Set<string>();

        currentIds.forEach((id) => {
            if (!prevIds.has(id)) {
                newIds.add(id);
            }
        });

        prevNodeIdsRef.current = currentIds;

        const isNewNode = (d: NodeData) => newIds.has(d.id);
        const isNewLink = (d: LinkData) => {
            const sid = typeof d.source === "string" ? d.source : (d.source as NodeData).id;
            const tid = typeof d.target === "string" ? d.target : (d.target as NodeData).id;
            return newIds.has(sid) || newIds.has(tid);
        };

        // 🔥 기존 simulation이 있으면 노드 위치 유지하면서 업데이트
        const nodeMap = new Map<string, NodeData>();
        if (simulationRef.current) {
            simulationRef.current.nodes().forEach(n => {
                nodeMap.set(n.id, n);
            });
        }

        // 새 노드 데이터에 기존 위치 복사 (있으면)
        data.nodes.forEach(node => {
            const existing = nodeMap.get(node.id);
            if (existing && existing.x !== undefined && existing.y !== undefined) {
                node.x = existing.x;
                node.y = existing.y;
                node.vx = existing.vx;
                node.vy = existing.vy;
                node.fx = existing.fx;
                node.fy = existing.fy;
            } else if (!node.x || !node.y) {
                // 새 노드: 중심 근처에 랜덤 배치
                node.x = width / 2 + (Math.random() - 0.5) * 100;
                node.y = height / 2 + (Math.random() - 0.5) * 100;
            }
        });

        console.log('📍 Node positions:', data.nodes.map(n => ({ id: n.id, x: n.x, y: n.y })));

        // --- Links 업데이트 (D3 data join pattern) ---
        const link = linkGroup
            .selectAll<SVGLineElement, LinkData>("line")
            .data(data.links, (d: LinkData) => {
                const sid = typeof d.source === "string" ? d.source : (d.source as NodeData).id;
                const tid = typeof d.target === "string" ? d.target : (d.target as NodeData).id;
                return `${sid}-${tid}-${d.predicate}`;
            });

        // EXIT
        link.exit()
            .transition()
            .duration(300)
            .attr("opacity", 0)
            .remove();

        // ENTER
        const linkEnter = link
            .enter()
            .append("line")
            .attr("stroke-width", 2)
            .attr("stroke", (d) => d.type === "temporal" ? TEMPORAL_COLOR : STATIC_COLOR)
            .attr("stroke-linecap", "round")
            .attr("opacity", 0)
            .attr("x1", (d) => (d.source as NodeData).x ?? width / 2)
            .attr("y1", (d) => (d.source as NodeData).y ?? height / 2)
            .attr("x2", (d) => (d.source as NodeData).x ?? width / 2)
            .attr("y2", (d) => (d.source as NodeData).y ?? height / 2);

        // 🔥 새 링크만 Draw + Fade-in
        linkEnter
            .filter((d) => isNewLink(d))
            .transition()
            .delay(100) // 🔥 딜레이 감소 (200 → 100)
            .duration(400) // 🔥 지속시간 감소 (650 → 400)
            .ease(d3.easeCubicOut)
            .attr("opacity", 0.9);

        // UPDATE
        const linkUpdate = linkEnter.merge(link);

        // --- Link labels 업데이트 ---
        const linkLabels = linkLabelGroup
            .selectAll<SVGTextElement, LinkData>("text")
            .data(data.links, (d: LinkData) => {
                const sid = typeof d.source === "string" ? d.source : (d.source as NodeData).id;
                const tid = typeof d.target === "string" ? d.target : (d.target as NodeData).id;
                return `${sid}-${tid}-${d.predicate}`;
            });

        linkLabels.exit().remove();

        const linkLabelsEnter = linkLabels
            .enter()
            .append("text")
            .text((d) => d.predicate ?? "")
            .attr("font-size", 11)
            .attr("fill", (d) => d.type === "temporal" ? TEMPORAL_COLOR : "#a0aec0")
            .attr("text-anchor", "middle")
            .style("pointer-events", "none")
            .style("opacity", 0);

        linkLabelsEnter
            .transition()
            .duration(400)
            .style("opacity", 1);

        const linkLabelsUpdate = linkLabelsEnter.merge(linkLabels);

        // --- Nodes 업데이트 ---
        const nodeGroup = nodeLabelGroup;
        const node = nodeGroup
            .selectAll<SVGCircleElement, NodeData>("circle")
            .data(data.nodes, (d: NodeData) => d.id);

        node.exit()
            .transition()
            .duration(300)
            .attr("r", 0)
            .style("opacity", 0)
            .remove();

        const nodeEnter = node
            .enter()
            .append("circle")
            .attr("r", (d) => isNewNode(d) ? 1 : nodeSize)
            .attr("fill", colorFunction)
            .attr("cx", (d) => d.x ?? width / 2)
            .attr("cy", (d) => d.y ?? height / 2)
            .style("cursor", "pointer")
            .style("opacity", (d) => isNewNode(d) ? 0 : 1)
            .style("filter", (d) => isNewNode(d) ? "blur(4px)" : "none")
            .on("click", (_, d: NodeData) => onNodeClick?.(d));

        nodeEnter
            .filter((d) => isNewNode(d))
            .transition()
            .duration(600) // 🔥 지속시간 감소 (950 → 600)
            .ease(d3.easeElasticOut.amplitude(1.1).period(0.3)) // 🔥 탄성 감소
            .attr("r", nodeSize)
            .style("opacity", 1)
            .style("filter", "blur(0px)");

        const nodeUpdate = nodeEnter.merge(node);

        // UPDATE 노드 색상
        nodeUpdate
            .transition()
            .duration(300)
            .attr("fill", colorFunction);

        // --- Node labels 업데이트 ---
        const label = nodeGroup
            .selectAll<SVGTextElement, NodeData>("text")
            .data(data.nodes, (d: NodeData) => d.id);

        label.exit().remove();

        const labelEnter = label
            .enter()
            .append("text")
            .text((d) => d.name ?? d.id)
            .attr("font-size", 12)
            .attr("dx", 12)
            .attr("dy", "0.35em")
            .attr("x", (d) => d.x ?? width / 2)
            .attr("y", (d) => d.y ?? height / 2)
            .style("pointer-events", "none")
            .style("fill", "#4a5568")
            .style("opacity", 0)
            .style("filter", "blur(4px)");

        labelEnter
            .transition()
            .duration(300) // 🔥 지속시간 감소 (400 → 300)
            .style("opacity", 1)
            .style("filter", "blur(0px)");

        const labelUpdate = labelEnter.merge(label);

        // --- Drag behavior ---
        const dragBehavior = d3
            .drag<SVGCircleElement, NodeData>()
            .on("start", (event, d) => {
                if (!event.active && simulationRef.current) {
                    simulationRef.current.alphaTarget(0.3).restart();
                }
                d.fx = d.x;
                d.fy = d.y;
            })
            .on("drag", (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on("end", (event, d) => {
                if (!event.active && simulationRef.current) {
                    simulationRef.current.alphaTarget(0);
                }
                d.fx = null;
                d.fy = null;
            });

        nodeUpdate.call(dragBehavior as any);

        // --- Simulation 업데이트 ---
        if (simulationRef.current) {
            // 🔥 기존 simulation을 재활용하고 새 데이터로 업데이트
            simulationRef.current.nodes(data.nodes);
            (simulationRef.current.force("link") as d3.ForceLink<NodeData, LinkData>)
                ?.links(data.links);

            // 🔥 alpha 값 조정: 큰 변화가 있을 때만 높게
            const nodeDiff = Math.abs(data.nodes.length - prevNodeIdsRef.current.size);
            const alphaValue = nodeDiff > 2 ? 0.5 : 0.2; // 노드가 많이 변경되면 0.5, 적으면 0.2

            console.log(`♻️ Reusing simulation with alpha=${alphaValue}`);
            simulationRef.current.alpha(alphaValue).restart();
        } else {
            // 🔥 처음에만 simulation 생성
            simulationRef.current = d3
                .forceSimulation<NodeData>(data.nodes)
                .force(
                    "link",
                    d3
                        .forceLink<NodeData, LinkData>(data.links)
                        .id((d) => d.id)
                        .distance(140)
                        .strength(0.5) // 🔥 링크 강도 낮춤 (더 부드럽게)
                )
                .force("charge", d3.forceManyBody().strength(-300)) // 🔥 반발력 낮춤
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collision", d3.forceCollide().radius(nodeSize * 2.5))
                .alphaDecay(0.01) // 🔥 더 천천히 감속 (0.02 → 0.01)
                .velocityDecay(0.3); // 🔥 관성 더 유지 (0.4 → 0.3)
        }

        simulationRef.current.on("tick", () => {
            // 🔥 부모에게 노드 위치 업데이트 전달
            if (onNodePositionUpdate) {
                data.nodes.forEach(node => {
                    if (node.x !== undefined && node.y !== undefined) {
                        onNodePositionUpdate(node.id, node.x, node.y, node.vx, node.vy);
                    }
                });
            }

            // 🔥 안전하게 위치 업데이트 (노드를 찾아서)
            linkUpdate
                .each(function (d) {
                    const sourceNode = data.nodes.find(n => n.id === (typeof d.source === "string" ? d.source : (d.source as NodeData).id));
                    const targetNode = data.nodes.find(n => n.id === (typeof d.target === "string" ? d.target : (d.target as NodeData).id));

                    if (sourceNode && targetNode) {
                        d3.select(this)
                            .attr("x1", sourceNode.x ?? 0)
                            .attr("y1", sourceNode.y ?? 0)
                            .attr("x2", targetNode.x ?? 0)
                            .attr("y2", targetNode.y ?? 0);
                    }
                });

            nodeUpdate
                .attr("cx", (d) => d.x ?? 0)
                .attr("cy", (d) => d.y ?? 0);

            labelUpdate
                .attr("x", (d) => d.x ?? 0)
                .attr("y", (d) => d.y ?? 0);

            linkLabelsUpdate
                .each(function (d) {
                    const sourceNode = data.nodes.find(n => n.id === (typeof d.source === "string" ? d.source : (d.source as NodeData).id));
                    const targetNode = data.nodes.find(n => n.id === (typeof d.target === "string" ? d.target : (d.target as NodeData).id));

                    if (sourceNode && targetNode) {
                        d3.select(this)
                            .attr("x", ((sourceNode.x ?? 0) + (targetNode.x ?? 0)) / 2)
                            .attr("y", ((sourceNode.y ?? 0) + (targetNode.y ?? 0)) / 2 - 6);
                    }
                });
        });
    }, [data, width, height, nodeSize, onNodeClick, colorFunction, onNodePositionUpdate]);

    useEffect(() => {
        if (!svgRef.current) return;

        drawGraph();

        return () => {
            if (simulationRef.current) {
                simulationRef.current.stop();
            }
        };
    }, [drawGraph]);

    // 🔥 width/height 변경시만 center force 업데이트 (drawGraph 재호출 안함)
    useEffect(() => {
        if (simulationRef.current && !isInitialRenderRef.current) {
            console.log('📐 Updating center force due to size change');
            simulationRef.current.force("center", d3.forceCenter(width / 2, height / 2));
            simulationRef.current.alpha(0.1).restart(); // 아주 약하게만 재시작
        }
    }, [width, height]);

    return <svg ref={svgRef} width={width} height={height} />;
};