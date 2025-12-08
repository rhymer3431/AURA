/* eslint-disable react-hooks/exhaustive-deps */
import * as d3 from "d3";
import React, { useCallback, useEffect, useRef } from "react";

export type NodeData = {
    id: string;
    name?: string;
    group?: number;
    x?: number;
    y?: number;
    fx?: number | null;
    fy?: number | null;
};

export type LinkData = {
    source: string | NodeData;
    target: string | NodeData;
    predicate?: string;
    type?: "static" | "temporal";
    confidence?: number;
};

export interface GraphData {
    nodes: NodeData[];
    links: LinkData[];
}

interface ForcedGraphProps {
    data: GraphData;
    width?: number;
    height?: number;
    nodeSize?: number;
    onNodeClick?: (node: NodeData) => void;
    colorFunction?: (node: NodeData) => string;
}

export const ForcedGraph: React.FC<ForcedGraphProps> = ({
    data,
    width = 800,
    height = 500,
    nodeSize = 12,
    onNodeClick,
    colorFunction = (d) => "#4E91F9",
}) => {
    const svgRef = useRef<SVGSVGElement | null>(null);
    const simulationRef =
        useRef<d3.Simulation<NodeData, LinkData> | null>(null);

    const drawGraph = useCallback(() => {
        const svg = d3.select(svgRef.current);
        svg.selectAll("*").remove();

        const zoomLayer = svg.append("g");

        // --- Links ---
        const link = zoomLayer
            .append("g")
            .attr("stroke-width", 2)
            .selectAll<SVGLineElement, LinkData>("line")
            .data(data.links)
            .enter()
            .append("line")
            .attr("stroke", (d) =>
                d.type === "temporal" ? "#805ad5" : "#4fd1c5"
            )
            .attr("stroke-dasharray", (d) =>
                d.type === "temporal" ? "5,5" : "0"
            )
            .attr("opacity", 0.7);

        // --- Link labels (relation text) ---
        const linkLabels = zoomLayer
            .append("g")
            .selectAll<SVGTextElement, LinkData>("text")
            .data(data.links)
            .enter()
            .append("text")
            .text((d) => d.predicate ?? "")
            .attr("font-size", 11)
            .attr("fill", (d) =>
                d.type === "temporal" ? "#805ad5" : "#a0aec0"
            )
            .attr("text-anchor", "middle")
            .style("pointer-events", "none");

        // --- Nodes ---
        const node = zoomLayer
            .append("g")
            .selectAll<SVGCircleElement, NodeData>("circle")
            .data(data.nodes)
            .enter()
            .append("circle")
            .attr("r", nodeSize)
            .attr("fill", colorFunction)
            .style("cursor", "pointer")
            .on("click", (_, d: NodeData) => onNodeClick?.(d));

        // --- Node labels ---
        const label = zoomLayer
            .append("g")
            .selectAll<SVGTextElement, NodeData>("text")
            .data(data.nodes)
            .enter()
            .append("text")
            .text((d) => d.name ?? d.id)
            .attr("font-size", 12)
            .attr("dx", 12)
            .attr("dy", "0.35em")
            .style("pointer-events", "none")
            .style("fill", "#4a5568");

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

        node.call(dragBehavior as any);

        // --- Force simulation ---
        simulationRef.current = d3
            .forceSimulation<NodeData>(data.nodes)
            .force(
                "link",
                d3
                    .forceLink<NodeData, LinkData>(data.links)
                    .id((d) => d.id)
                    .distance(140)
            )
            .force("charge", d3.forceManyBody().strength(-420))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .on("tick", () => {
                link
                    .attr("x1", (d) => (d.source as NodeData).x ?? 0)
                    .attr("y1", (d) => (d.source as NodeData).y ?? 0)
                    .attr("x2", (d) => (d.target as NodeData).x ?? 0)
                    .attr("y2", (d) => (d.target as NodeData).y ?? 0);

                node
                    .attr("cx", (d) => d.x ?? 0)
                    .attr("cy", (d) => d.y ?? 0);

                label
                    .attr("x", (d) => d.x ?? 0)
                    .attr("y", (d) => d.y ?? 0);

                linkLabels
                    .attr("x", (d) => {
                        const sx = (d.source as NodeData).x ?? 0;
                        const tx = (d.target as NodeData).x ?? 0;
                        return (sx + tx) / 2;
                    })
                    .attr("y", (d) => {
                        const sy = (d.source as NodeData).y ?? 0;
                        const ty = (d.target as NodeData).y ?? 0;
                        return (sy + ty) / 2 - 6;
                    });
            });

        // --- Zoom behavior ---
        const zoomBehavior = d3
            .zoom<SVGSVGElement, unknown>()
            .on("zoom", (event) => {
                zoomLayer.attr("transform", event.transform);
            });

        svg.call(zoomBehavior as any);
    }, [data, width, height, nodeSize, onNodeClick, colorFunction]);

    useEffect(() => {
        if (!svgRef.current) return;
        drawGraph();
        return () => {
            if (simulationRef.current) {
                simulationRef.current.stop();
            }
        };
    }, [drawGraph]);

    return <svg ref={svgRef} width={width} height={height} />;
};
