import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { FileText, TrendingUp } from "lucide-react";

interface ReportViewProps {
  paper: string;
  analysis: string;
}

export const ReportView = ({ paper, analysis }: ReportViewProps) => {
  const [activeTab, setActiveTab] = useState("paper");

  return (
    <Card className="p-6 bg-card border-border">
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full max-w-md mx-auto grid-cols-2 mb-6 bg-secondary">
          <TabsTrigger 
            value="paper" 
            className="data-[state=active]:bg-gradient-primary data-[state=active]:text-primary-foreground"
          >
            <FileText className="w-4 h-4 mr-2" />
            Generated Paper
          </TabsTrigger>
          <TabsTrigger 
            value="analysis"
            className="data-[state=active]:bg-gradient-primary data-[state=active]:text-primary-foreground"
          >
            <TrendingUp className="w-4 h-4 mr-2" />
            Industry Analysis
          </TabsTrigger>
        </TabsList>

        <TabsContent value="paper" className="mt-0">
          <div className="prose prose-invert prose-headings:text-foreground prose-p:text-foreground prose-strong:text-foreground prose-code:text-primary prose-pre:bg-secondary max-w-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {paper}
            </ReactMarkdown>
          </div>
        </TabsContent>

        <TabsContent value="analysis" className="mt-0">
          <div className="prose prose-invert prose-headings:text-foreground prose-p:text-foreground prose-strong:text-foreground prose-code:text-primary prose-pre:bg-secondary max-w-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {analysis}
            </ReactMarkdown>
          </div>
        </TabsContent>
      </Tabs>
    </Card>
  );
};
