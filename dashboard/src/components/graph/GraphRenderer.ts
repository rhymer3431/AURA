import * as d3 from "d3";
import type { GraphData, LinkData, NodeData } from "../../domain/types/GraphTypes";
import { createDragBehavior, createZoomBehavior } from "./GraphBehaviors";
import { createSimulation } from "./GraphSimulation";

const BASE_RADIUS = 18;
const APPROX_CHAR_WIDTH = 6;
const TEXT_PADDING = 6;

const COLOR_BG = "#ffffff";
const COLOR_NEUTRAL_100 = "#F8FAFC";
const COLOR_NEUTRAL_300 = "#DFE5EC";
const COLOR_TEXT = "#64748B";
const COLOR_LINK = "#DFE5EC";
const COLOR_LINK_TEMPORAL = "#94A3B8";
const COLOR_LINK_LABEL = "#94A3B8";

const getLabel = (d: NodeData) => d.name ?? String(d.id);

const getRadius = (d: NodeData, nodeSize: number) => {
    const label = getLabel(d);
    const approxTextWidth = label.length * APPROX_CHAR_WIDTH;
    const textBasedRadius = approxTextWidth / 2 + TEXT_PADDING;
    return Math.max(BASE_RADIUS, nodeSize, textBasedRadius);
};

type GraphController = {
    update: (
        nextData: GraphData,
        opts?: {
            width?: number;
            height?: number;
            nodeSize?: number;
            onNodeClick?: (node: NodeData) => void;
            colorFunction?: (node: NodeData) => string;
        }
    ) => void;
    stop: () => void;
};

export const renderGraph = (
    svgRef: SVGSVGElement,
    data: GraphData,
    width: number,
    height: number,
    nodeSize: number,
    onNodeClick?: (node: NodeData) => void,
    colorFunction?: (node: NodeData) => string
): GraphController => {
    const svg = d3.select(svgRef);
    svg.selectAll("*").remove();

    const state = {
        width,
        height,
        nodeSize,
        onNodeClick,
        colorFunction,
    };

    svg.attr("width", width).attr("height", height).style("background", COLOR_BG);

    const frame = svg
        .append("rect")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", width)
        .attr("height", height)
        .attr("fill", "none")
        .attr("stroke", COLOR_NEUTRAL_300)
        .attr("rx", 5)
        .attr("ry", 5);

    const zoomLayer = svg.append("g");

    const linkGroup = zoomLayer.append("g").attr("stroke-width", 1.5);
    const nodeLayer = zoomLayer.append("g");
    const labelLayer = zoomLayer.append("g");

    let linkSelection = linkGroup.selectAll<SVGLineElement, LinkData>("line");
    let nodeSelection = nodeLayer.selectAll<SVGGElement, NodeData>("g");
    let linkLabelSelection = labelLayer.selectAll<SVGTextElement, LinkData>("text");

    const simulation = createSimulation(data.nodes, data.links, width, height, () => {
        nodeSelection.attr("transform", (d) => `translate(${d.x ?? 0}, ${d.y ?? 0})`);

        linkSelection
            .attr("x1", (d) => (d.source as NodeData).x ?? 0)
            .attr("y1", (d) => (d.source as NodeData).y ?? 0)
            .attr("x2", (d) => (d.target as NodeData).x ?? 0)
            .attr("y2", (d) => (d.target as NodeData).y ?? 0);

        linkLabelSelection
            .attr("x", (d) => {
                const sx = (d.source as NodeData).x ?? 0;
                const tx = (d.target as NodeData).x ?? 0;
                return (sx + tx) / 2;
            })
            .attr("y", (d) => {
                const sy = (d.source as NodeData).y ?? 0;
                const ty = (d.target as NodeData).y ?? 0;
                return (sy + ty) / 2 - 4;
            });
    });

    nodeLayer.call(createDragBehavior(simulation) as any);
    svg.call(createZoomBehavior(zoomLayer, width, height) as any);

    const applyNodeStyles = (
        sel: d3.Selection<SVGGElement, NodeData, SVGGElement | null, unknown>
    ) => {
        sel.select<SVGCircleElement>("circle")
            .attr("r", (d) => getRadius(d, state.nodeSize))
            .attr("fill", COLOR_BG)
            .attr("stroke", (d) => state.colorFunction?.(d) ?? COLOR_NEUTRAL_300)
            .attr("stroke-width", 1.2);

        sel.select<SVGTextElement>("text")
            .text((d) => getLabel(d))
            .attr("x", 0)
            .attr("y", 4)
            .attr("font-size", 12)
            .attr("font-weight", 500)
            .attr("fill", COLOR_TEXT)
            .attr("text-anchor", "middle")
            .style("font-family", "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif")
            .style("pointer-events", "none");
    };

    const update = (
        nextData: GraphData,
        opts?: {
            width?: number;
            height?: number;
            nodeSize?: number;
            onNodeClick?: (node: NodeData) => void;
            colorFunction?: (node: NodeData) => string;
        }
    ) => {
        if (opts?.width !== undefined && opts?.height !== undefined) {
            state.width = opts.width;
            state.height = opts.height;
            svg.attr("width", state.width).attr("height", state.height);
            frame.attr("width", state.width).attr("height", state.height);
            simulation.force("center", d3.forceCenter(state.width / 2, state.height / 2));
            svg.call(createZoomBehavior(zoomLayer, state.width, state.height) as any);
        }

        if (opts?.nodeSize !== undefined) state.nodeSize = opts.nodeSize;
        if (opts?.onNodeClick) state.onNodeClick = opts.onNodeClick;
        if (opts?.colorFunction) state.colorFunction = opts.colorFunction;

        const previousPositions = new Map(nodeSelection.data().map((n) => [String(n.id), n]));

        const nodesWithPositions = nextData.nodes.map((n) => {
            const prev = previousPositions.get(String(n.id));
            if (!prev) return n;
            return { ...n, x: prev.x, y: prev.y, vx: prev.vx, vy: prev.vy };
        });

        linkSelection = linkSelection
            .data(nextData.links, (d: any) => `${d.source}-${d.target}-${d.predicate ?? ""}`)
            .join(
                (enter) =>
                    enter
                        .append("line")
                        .attr("stroke", (d) => (d.type === "temporal" ? COLOR_LINK_TEMPORAL : COLOR_LINK))
                        .attr("stroke-width", 1.5)
                        .attr("opacity", 0)
                        .call((sel) => sel.transition().duration(250).attr("opacity", 0.9)),
                (updateSel) => updateSel,
                (exit) => exit.transition().duration(200).attr("opacity", 0).remove()
            );

        nodeSelection = nodeSelection
            .data(nodesWithPositions, (d: any) => String(d.id))
            .join(
                (enter) => {
                    const g = enter
                        .append("g")
                        .style("cursor", "pointer")
                        .style("opacity", 0)
                        .on("click", (_, d) => state.onNodeClick?.(d));

                    g.append("circle");
                    g.append("text");
                    applyNodeStyles(g);

                    g.on("mouseover", function () {
                        d3.select(this)
                            .select<SVGCircleElement>("circle")
                            .transition()
                            .duration(120)
                            .attr("stroke-width", 2)
                            .attr("fill", COLOR_NEUTRAL_100);
                    }).on("mouseout", function () {
                        d3.select(this)
                            .select<SVGCircleElement>("circle")
                            .transition()
                            .duration(120)
                            .attr("stroke-width", 1.2)
                            .attr("fill", COLOR_BG);
                    });

                    g.transition().duration(250).style("opacity", 1);
                    return g;
                },
                (updateSel) => {
                    updateSel.on("click", (_, d) => state.onNodeClick?.(d));
                    applyNodeStyles(updateSel);
                    return updateSel;
                },
                (exit) => exit.transition().duration(200).style("opacity", 0).remove()
            );

        linkLabelSelection = linkLabelSelection
            .data(nextData.links, (d: any) => `${d.source}-${d.target}-${d.predicate ?? ""}`)
            .join(
                (enter) =>
                    enter
                        .append("text")
                        .text((d) => d.predicate ?? "")
                        .attr("font-size", 10)
                        .attr("fill", COLOR_LINK_LABEL)
                        .attr("text-anchor", "middle")
                        .style("pointer-events", "none")
                        .style("opacity", 0)
                        .call((sel) => sel.transition().duration(250).style("opacity", 1)),
                (updateSel) => updateSel.text((d) => d.predicate ?? ""),
                (exit) => exit.transition().duration(200).style("opacity", 0).remove()
            );

        simulation.nodes(nodesWithPositions);
        const linkForce = simulation.force("link") as d3.ForceLink<NodeData, LinkData>;
        linkForce.links(nextData.links);

        simulation.alpha(0.6).restart();
    };

    update(data);

    return {
        update,
        stop: () => simulation.stop(),
    };
};
