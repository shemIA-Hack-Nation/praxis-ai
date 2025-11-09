import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { FileCode, FileText, BarChart3, Image, Play } from "lucide-react";
import TerminalOutput from "./AnsiOutput";

interface NotebookCell {
  type: string;
  content: any;
  execution_count?: number;
  source?: string;
  outputs?: any[];
}

interface NotebookSummary {
  total_elements: number;
  code_cells: number;
  markdown_cells: number;
  output_cells: number;
}

interface NotebookViewProps {
  cells: NotebookCell[];
  summary: NotebookSummary;
  message: string;
}

const getCellIcon = (type: string) => {
  switch (type) {
    case 'code':
      return <FileCode className="w-4 h-4" />;
    case 'markdown':
      return <FileText className="w-4 h-4" />;
    case 'output_text':
      return <Play className="w-4 h-4" />;
    case 'output_plot':
      return <Image className="w-4 h-4" />;
    default:
      return <BarChart3 className="w-4 h-4" />;
  }
};

const getCellBadgeColor = (type: string) => {
  switch (type) {
    case 'code':
      return 'bg-blue-500/20 text-blue-300 border-blue-500/30';
    case 'markdown':
      return 'bg-green-500/20 text-green-300 border-green-500/30';
    case 'output_text':
      return 'bg-purple-500/20 text-purple-300 border-purple-500/30';
    case 'output_plot':
      return 'bg-orange-500/20 text-orange-300 border-orange-500/30';
    default:
      return 'bg-gray-500/20 text-gray-300 border-gray-500/30';
  }
};

export const NotebookView = ({ cells, summary, message }: NotebookViewProps) => {
  const [activeTab, setActiveTab] = useState("overview");

  const renderCellContent = (cell: NotebookCell) => {
    if (cell.type === 'code') {
      const codeContent = cell.source || cell.content || '';
      return (
        <div className="space-y-2">
          {codeContent && (
            <pre className="bg-gray-900/50 p-3 rounded text-sm overflow-x-auto">
              <code className="text-blue-300">{codeContent}</code>
            </pre>
          )}
          {cell.execution_count && (
            <div className="text-xs text-muted-foreground">
              Execution count: {cell.execution_count}
            </div>
          )}
        </div>
      );
    }

    if (cell.type === 'markdown') {
      const markdownContent = cell.source || cell.content || '';
      return (
        <div className="bg-gray-900/30 p-3 rounded">
          <pre className="text-sm text-gray-300 whitespace-pre-wrap">{markdownContent}</pre>
        </div>
      );
    }

    if (cell.type === 'output_text') {
      const textContent = typeof cell.content === 'string' ? cell.content : JSON.stringify(cell.content, null, 2);
      return (
        <div className="bg-gray-900/40 p-2 rounded border border-gray-600/30">
          <TerminalOutput 
            text={textContent} 
            className=""
          />
        </div>
      );
    }

    if (cell.type === 'output_plot') {
      return (
        <div className="bg-orange-900/20 p-3 rounded border border-orange-500/20">
          {typeof cell.content === 'object' && cell.content?.image_path ? (
            <div className="space-y-2">
              <img 
                src={`http://localhost:8000/images/${cell.content.notebook_id}/${cell.content.image_path}`}
                alt="Plot output" 
                className="max-w-full h-auto rounded"
                onError={(e) => {
                  console.error('Failed to load image:', e.currentTarget.src);
                  e.currentTarget.style.display = 'none';
                }}
              />
              <div className="text-xs text-orange-300">
                Notebook ID: {cell.content.notebook_id}
              </div>
              <div className="text-xs text-gray-400">
                Image path: {cell.content.image_path}
              </div>
            </div>
          ) : (
            <pre className="text-sm text-orange-200">
              {typeof cell.content === 'string' ? cell.content : JSON.stringify(cell.content, null, 2)}
            </pre>
          )}
        </div>
      );
    }

    return (
      <div className="bg-gray-800/50 p-3 rounded">
        <pre className="text-sm text-gray-300">
          {typeof cell.content === 'string' ? cell.content : JSON.stringify(cell.content, null, 2)}
        </pre>
      </div>
    );
  };

  return (
    <Card className="p-6 bg-card border-border">
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full max-w-lg mx-auto grid-cols-3 mb-6 bg-secondary">
          <TabsTrigger 
            value="overview" 
            className="data-[state=active]:bg-gradient-primary data-[state=active]:text-primary-foreground"
          >
            <BarChart3 className="w-4 h-4 mr-2" />
            Overview
          </TabsTrigger>
          <TabsTrigger 
            value="cells"
            className="data-[state=active]:bg-gradient-primary data-[state=active]:text-primary-foreground"
          >
            <FileCode className="w-4 h-4 mr-2" />
            Cells
          </TabsTrigger>
          <TabsTrigger 
            value="outputs"
            className="data-[state=active]:bg-gradient-primary data-[state=active]:text-primary-foreground"
          >
            <Image className="w-4 h-4 mr-2" />
            Outputs
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="mt-0">
          <div className="space-y-6">
            <div className="text-center">
              <h3 className="text-lg font-semibold text-foreground mb-2">{message}</h3>
              <div className="flex justify-center gap-4 text-sm text-muted-foreground">
                <span>Total: {summary.total_elements}</span>
                <span>•</span>
                <span>Code: {summary.code_cells}</span>
                <span>•</span>
                <span>Markdown: {summary.markdown_cells}</span>
                <span>•</span>
                <span>Outputs: {summary.output_cells}</span>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Card className="p-4 text-center">
                <FileCode className="w-8 h-8 mx-auto mb-2 text-blue-400" />
                <div className="text-2xl font-bold text-foreground">{summary.code_cells}</div>
                <div className="text-sm text-muted-foreground">Code Cells</div>
              </Card>
              <Card className="p-4 text-center">
                <FileText className="w-8 h-8 mx-auto mb-2 text-green-400" />
                <div className="text-2xl font-bold text-foreground">{summary.markdown_cells}</div>
                <div className="text-sm text-muted-foreground">Markdown</div>
              </Card>
              <Card className="p-4 text-center">
                <BarChart3 className="w-8 h-8 mx-auto mb-2 text-purple-400" />
                <div className="text-2xl font-bold text-foreground">{summary.output_cells}</div>
                <div className="text-sm text-muted-foreground">Outputs</div>
              </Card>
              <Card className="p-4 text-center">
                <Play className="w-8 h-8 mx-auto mb-2 text-orange-400" />
                <div className="text-2xl font-bold text-foreground">{summary.total_elements}</div>
                <div className="text-sm text-muted-foreground">Total</div>
              </Card>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="cells" className="mt-0">
          <div className="space-y-4">
            <Accordion type="single" collapsible className="space-y-2">
              {cells.map((cell, index) => (
                <AccordionItem key={index} value={`cell-${index}`} className="border border-border rounded-lg">
                  <AccordionTrigger className="px-4 py-3 hover:no-underline">
                    <div className="flex items-center gap-3 text-left">
                      {getCellIcon(cell.type)}
                      <span className="font-medium">Cell {index + 1}</span>
                      <Badge className={`${getCellBadgeColor(cell.type)} text-xs`}>
                        {cell.type}
                      </Badge>
                      {cell.execution_count && (
                        <Badge variant="outline" className="text-xs">
                          [{cell.execution_count}]
                        </Badge>
                      )}
                    </div>
                  </AccordionTrigger>
                  <AccordionContent className="px-4 pb-4">
                    {renderCellContent(cell)}
                  </AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>
          </div>
        </TabsContent>

        <TabsContent value="outputs" className="mt-0">
          <div className="space-y-4">
            <Accordion type="single" collapsible className="space-y-2">
              {cells
                .filter(cell => cell.type.startsWith('output_'))
                .map((cell, index) => (
                  <AccordionItem key={index} value={`output-${index}`} className="border border-border rounded-lg">
                    <AccordionTrigger className="px-4 py-3 hover:no-underline">
                      <div className="flex items-center gap-3 text-left">
                        {getCellIcon(cell.type)}
                        <span className="font-medium">Output {index + 1}</span>
                        <Badge className={`${getCellBadgeColor(cell.type)} text-xs`}>
                          {cell.type}
                        </Badge>
                      </div>
                    </AccordionTrigger>
                    <AccordionContent className="px-4 pb-4">
                      {renderCellContent(cell)}
                    </AccordionContent>
                  </AccordionItem>
                ))}
            </Accordion>
            {cells.filter(cell => cell.type.startsWith('output_')).length === 0 && (
              <div className="text-center text-muted-foreground py-8">
                No outputs found in this notebook.
              </div>
            )}
          </div>
        </TabsContent>
      </Tabs>
    </Card>
  );
};