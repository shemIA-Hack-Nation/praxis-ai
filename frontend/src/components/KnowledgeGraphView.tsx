/**
 * D3.js Knowledge Graph Visualization Component
 * Displays relationships between papers, concepts, and methods
 */
import { useEffect, useRef } from "react";
import * as d3 from "d3";
import { Card } from "@/components/ui/card";

interface KnowledgeGraphNode {
  id: number;
  name: string;
  type: string;
  year?: number;
  novelty_score?: number;
  group: number;
  title: string;
  abstract?: string;
}

interface KnowledgeGraphLink {
  source: number | KnowledgeGraphNode;
  target: number | KnowledgeGraphNode;
  type: string;
  value: number;
  similarity?: number;
}

interface KnowledgeGraphData {
  nodes: KnowledgeGraphNode[];
  links: KnowledgeGraphLink[];
  stats: {
    total_nodes: number;
    total_links: number;
    target_paper: string;
    retrieved_papers: number;
    novelty_score: number;
  };
}

interface KnowledgeGraphViewProps {
  data: KnowledgeGraphData;
}

export const KnowledgeGraphView = ({ data }: KnowledgeGraphViewProps) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!data || !data.nodes || !data.links || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove(); // Clear previous render

    const width = containerRef.current?.clientWidth || 800;
    const height = 600;

    svg.attr("width", width).attr("height", height);

    // Create container group for zoom/pan
    const container = svg.append("g").attr("class", "zoom-container");

    // Set up zoom behavior
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4]) // Min and max zoom levels
      .on("zoom", (event) => {
        container.attr("transform", event.transform);
      });

    svg.call(zoom);

    // Create simulation
    const simulation = d3
      .forceSimulation<KnowledgeGraphNode>(data.nodes as any)
      .force(
        "link",
        d3
          .forceLink<KnowledgeGraphNode, KnowledgeGraphLink>(data.links as any)
          .id((d: any) => d.id)
          .distance((d: any) => {
            // Distance based on relationship type
            if (d.type === "SIMILAR_TO") return 100;
            if (d.type === "RELATED_TO") return 150;
            if (d.type === "DISCUSSES" || d.type === "USES") return 80;
            return 120;
          })
          .strength((d: any) => {
            // Strength based on similarity
            if (d.similarity) return d.similarity / 10;
            return d.value || 0.5;
          })
      )
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(30));

    // Create links
    const link = container
      .append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(data.links)
      .enter()
      .append("line")
      .attr("stroke", (d) => {
        if (d.type === "SIMILAR_TO") return "#3b82f6"; // Blue for similarity
        if (d.type === "RELATED_TO") return "#10b981"; // Green for related
        if (d.type === "DISCUSSES") return "#f59e0b"; // Orange for concepts
        if (d.type === "USES") return "#8b5cf6"; // Purple for methods
        return "#6b7280"; // Gray default
      })
      .attr("stroke-width", (d) => Math.sqrt(d.value || 1) * 2)
      .attr("stroke-opacity", 0.6);

    // Create nodes
    const node = container
      .append("g")
      .attr("class", "nodes")
      .selectAll("circle")
      .data(data.nodes)
      .enter()
      .append("circle")
      .attr("r", (d) => {
        if (d.type === "TargetPaper") return 15;
        if (d.type === "Paper") return 10;
        return 6;
      })
      .attr("fill", (d) => {
        if (d.type === "TargetPaper") return "#ef4444"; // Red for target
        if (d.type === "Paper") return "#3b82f6"; // Blue for papers
        if (d.type === "Concept") return "#10b981"; // Green for concepts
        if (d.type === "Method") return "#8b5cf6"; // Purple for methods
        return "#6b7280";
      })
      .attr("stroke", "#fff")
      .attr("stroke-width", 2)
      .call(
        d3
          .drag<SVGCircleElement, KnowledgeGraphNode>()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended)
      );

    // Add labels
    const label = container
      .append("g")
      .attr("class", "labels")
      .selectAll("text")
      .data(data.nodes)
      .enter()
      .append("text")
      .text((d) => d.name)
      .attr("font-size", (d) => (d.type === "TargetPaper" ? "12px" : "10px"))
      .attr("fill", "#fff")
      .attr("dx", 12)
      .attr("dy", 4);

    // Add tooltips
    const tooltip = d3
      .select("body")
      .append("div")
      .attr("class", "tooltip")
      .style("opacity", 0)
      .style("position", "absolute")
      .style("background", "rgba(0, 0, 0, 0.9)")
      .style("color", "#fff")
      .style("padding", "8px")
      .style("border-radius", "4px")
      .style("pointer-events", "none")
      .style("z-index", "1000")
      .style("font-size", "12px")
      .style("max-width", "300px");

    node
      .on("mouseover", (event, d) => {
        tooltip.transition().duration(200).style("opacity", 0.9);
        tooltip
          .html(
            `<strong>${d.title}</strong><br/>` +
              (d.type ? `Type: ${d.type}<br/>` : "") +
              (d.year ? `Year: ${d.year}<br/>` : "") +
              (d.novelty_score ? `Novelty: ${d.novelty_score}/10<br/>` : "") +
              (d.abstract ? `Abstract: ${d.abstract.substring(0, 100)}...` : "")
          )
          .style("left", event.pageX + 10 + "px")
          .style("top", event.pageY - 10 + "px");
      })
      .on("mouseout", () => {
        tooltip.transition().duration(200).style("opacity", 0);
      });

    // Update positions on simulation tick
    simulation.on("tick", () => {
      link
        .attr("x1", (d: any) => (d.source as any).x)
        .attr("y1", (d: any) => (d.source as any).y)
        .attr("x2", (d: any) => (d.target as any).x)
        .attr("y2", (d: any) => (d.target as any).y);

      node.attr("cx", (d: any) => d.x).attr("cy", (d: any) => d.y);

      label.attr("x", (d: any) => d.x).attr("y", (d: any) => d.y);
    });

    // Drag functions (for node dragging)
    function dragstarted(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event: any, d: any) {
      // Apply zoom transform to drag coordinates
      const transform = d3.zoomTransform(svg.node()!);
      d.fx = (event.x - transform.x) / transform.k;
      d.fy = (event.y - transform.y) / transform.k;
    }

    function dragended(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }

    // Reset zoom button handler
    const resetZoom = () => {
      svg.transition()
        .duration(750)
        .call(
          zoom.transform,
          d3.zoomIdentity.translate(0, 0).scale(1)
        );
    };

    // Add reset zoom button
    const resetButton = svg
      .append("g")
      .attr("class", "reset-zoom")
      .style("cursor", "pointer")
      .on("click", resetZoom);

    resetButton
      .append("rect")
      .attr("x", 10)
      .attr("y", 10)
      .attr("width", 100)
      .attr("height", 30)
      .attr("fill", "rgba(0, 0, 0, 0.7)")
      .attr("rx", 4);

    resetButton
      .append("text")
      .attr("x", 60)
      .attr("y", 28)
      .attr("text-anchor", "middle")
      .attr("fill", "#fff")
      .attr("font-size", "12px")
      .text("Reset Zoom");

    // Add zoom controls
    const zoomControls = svg
      .append("g")
      .attr("class", "zoom-controls")
      .attr("transform", `translate(${width - 60}, 10)`);

    // Zoom in button
    const zoomIn = zoomControls
      .append("g")
      .style("cursor", "pointer")
      .on("click", () => {
        svg.transition().call(zoom.scaleBy, 1.5);
      });

    zoomIn
      .append("rect")
      .attr("width", 40)
      .attr("height", 30)
      .attr("fill", "rgba(0, 0, 0, 0.7)")
      .attr("rx", 4);

    zoomIn
      .append("text")
      .attr("x", 20)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .attr("fill", "#fff")
      .attr("font-size", "16px")
      .text("+");

    // Zoom out button
    const zoomOut = zoomControls
      .append("g")
      .attr("transform", "translate(0, 35)")
      .style("cursor", "pointer")
      .on("click", () => {
        svg.transition().call(zoom.scaleBy, 0.75);
      });

    zoomOut
      .append("rect")
      .attr("width", 40)
      .attr("height", 30)
      .attr("fill", "rgba(0, 0, 0, 0.7)")
      .attr("rx", 4);

    zoomOut
      .append("text")
      .attr("x", 20)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .attr("fill", "#fff")
      .attr("font-size", "16px")
      .text("−");

    // Cleanup
    return () => {
      tooltip.remove();
      simulation.stop();
    };
  }, [data]);

  if (!data || !data.nodes || data.nodes.length === 0) {
    return (
      <Card className="p-6">
        <p className="text-muted-foreground">No knowledge graph data available</p>
      </Card>
    );
  }

  return (
    <Card className="p-6">
      <div className="mb-4">
        <h3 className="text-lg font-semibold mb-2">Knowledge Graph</h3>
        <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
          <div>
            <span className="inline-block w-3 h-3 rounded-full bg-red-500 mr-2"></span>
            Target Paper
          </div>
          <div>
            <span className="inline-block w-3 h-3 rounded-full bg-blue-500 mr-2"></span>
            Retrieved Papers
          </div>
          <div>
            <span className="inline-block w-3 h-3 rounded-full bg-green-500 mr-2"></span>
            Concepts
          </div>
          <div>
            <span className="inline-block w-3 h-3 rounded-full bg-purple-500 mr-2"></span>
            Methods
          </div>
        </div>
        <div className="mt-2 text-sm">
          <p>
            <strong>Nodes:</strong> {data.stats.total_nodes} | <strong>Links:</strong>{" "}
            {data.stats.total_links} | <strong>Novelty Score:</strong> {data.stats.novelty_score}/10
          </p>
        </div>
      </div>
      <div className="mb-2 text-xs text-muted-foreground">
        <p>💡 Use mouse wheel to zoom, drag background to pan, drag nodes to reposition</p>
      </div>
      <div ref={containerRef} className="w-full border rounded-lg bg-secondary/50 overflow-hidden cursor-move">
        <svg ref={svgRef} className="w-full h-full"></svg>
      </div>
    </Card>
  );
};

