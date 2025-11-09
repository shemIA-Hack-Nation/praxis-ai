import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TrendingUp, Network, Star } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { KnowledgeGraphView } from "./KnowledgeGraphView";

interface KnowledgeGraphData {
  nodes: any[];
  links: any[];
  stats: {
    total_nodes: number;
    total_links: number;
    target_paper: string;
    retrieved_papers: number;
    novelty_score: number;
  };
}

interface ReportViewProps {
  novelty_analysis: string;
  knowledge_graph: KnowledgeGraphData;
}

export const ReportView = ({ novelty_analysis, knowledge_graph }: ReportViewProps) => {
  const [activeTab, setActiveTab] = useState("analysis");
  const noveltyScore = knowledge_graph?.stats?.novelty_score || 0;

  return (
    <div className="space-y-6">
      {/* Initial Novelty Score Banner */}
      <Card className="p-6 bg-gradient-to-r from-primary/10 via-primary/5 to-transparent border-2 border-primary/20">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-primary/20 rounded-full">
              <Star className="w-6 h-6 text-primary fill-primary" />
            </div>
            <div>
              <div className="text-sm text-muted-foreground mb-1">Initial Novelty Score</div>
              <div className="text-3xl font-bold text-foreground">
                {noveltyScore.toFixed(1)}<span className="text-lg text-muted-foreground">/10</span>
              </div>
            </div>
          </div>
          <Badge variant="outline" className="text-lg px-4 py-2 bg-background/50">
            {noveltyScore >= 8 ? "Very High Novelty" :
             noveltyScore >= 6 ? "High Novelty" :
             noveltyScore >= 4 ? "Medium Novelty" :
             noveltyScore >= 2 ? "Low Novelty" : "Very Low Novelty"}
          </Badge>
        </div>
      </Card>

      <Card className="p-6 bg-card border-border">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full max-w-md mx-auto grid-cols-2 mb-6 bg-secondary">
            <TabsTrigger 
              value="analysis" 
              className="data-[state=active]:bg-gradient-primary data-[state=active]:text-primary-foreground"
            >
              <TrendingUp className="w-4 h-4 mr-2" />
              Novelty Analysis
            </TabsTrigger>
            <TabsTrigger 
              value="graph"
              className="data-[state=active]:bg-gradient-primary data-[state=active]:text-primary-foreground"
            >
              <Network className="w-4 h-4 mr-2" />
              Knowledge Graph
            </TabsTrigger>
          </TabsList>

          <TabsContent value="analysis" className="mt-0">
            <div className="prose prose-invert prose-headings:text-foreground prose-p:text-foreground prose-strong:text-foreground prose-code:text-primary prose-pre:bg-secondary max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {novelty_analysis}
              </ReactMarkdown>
            </div>
          </TabsContent>

          <TabsContent value="graph" className="mt-0">
            <KnowledgeGraphView data={knowledge_graph} />
          </TabsContent>
        </Tabs>
      </Card>
    </div>
  );
};
