import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { FileCode, FileText, Image, Play, Loader2, CheckCircle, ArrowRight } from "lucide-react";
import TerminalOutput from "./AnsiOutput";

interface StreamingCell {
  type: string;
  content: any;
  execution_count?: number;
  source?: string;
}

interface StreamingUpdate {
  status: 'starting' | 'parsing' | 'completed' | 'orchestrating' | 'finished' | 'error';
  message: string;
  progress: number;
  cell?: StreamingCell;
  cell_index?: number;
  total_cells?: number;
  cells?: StreamingCell[];
  summary?: any;
  orchestrator_result?: any;
  error?: string;
}

interface StreamingNotebookProps {
  file: File;
  onComplete: (result: any) => void;
  onError: (error: string) => void;
}

const getCellIcon = (type: string) => {
  switch (type) {
    case 'code': return <FileCode className="w-4 h-4" />;
    case 'markdown': return <FileText className="w-4 h-4" />;
    case 'output_text': return <Play className="w-4 h-4" />;
    case 'output_plot': return <Image className="w-4 h-4" />;
    default: return <FileCode className="w-4 h-4" />;
  }
};

const getCellBadgeColor = (type: string) => {
  switch (type) {
    case 'code': return 'bg-blue-500/20 text-blue-300 border-blue-500/30';
    case 'markdown': return 'bg-green-500/20 text-green-300 border-green-500/30';
    case 'output_text': return 'bg-purple-500/20 text-purple-300 border-purple-500/30';
    case 'output_plot': return 'bg-orange-500/20 text-orange-300 border-orange-500/30';
    default: return 'bg-gray-500/20 text-gray-300 border-gray-500/30';
  }
};

const renderCellContent = (cell: StreamingCell) => {
  if (cell.type === 'code') {
    const codeContent = cell.source || cell.content || '';
    return (
      <pre className="bg-gray-900/50 p-3 rounded text-sm overflow-x-auto">
        <code className="text-blue-300">{codeContent}</code>
      </pre>
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
        <TerminalOutput text={textContent} className="" />
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

export const StreamingNotebook = ({ file, onComplete, onError }: StreamingNotebookProps) => {
  const [currentUpdate, setCurrentUpdate] = useState<StreamingUpdate | null>(null);
  const [processedCells, setProcessedCells] = useState<StreamingCell[]>([]);
  const [isOrchestratorPhase, setIsOrchestratorPhase] = useState(false);

  useEffect(() => {
    const processNotebook = async () => {
      try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('http://localhost:8000/api/v1/upload_notebook_stream/', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body?.getReader();
        const decoder = new TextDecoder();

        if (!reader) {
          throw new Error('No response body reader available');
        }

        while (true) {
          const { done, value } = await reader.read();
          
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.substring(6));
                setCurrentUpdate(data);

                if (data.cell) {
                  setProcessedCells(prev => [...prev, data.cell]);
                }

                if (data.status === 'completed') {
                  // Start orchestrator phase
                  setIsOrchestratorPhase(true);
                  
                  // Simulate orchestrator processing
                  setTimeout(() => {
                    const orchestratorUpdate: StreamingUpdate = {
                      status: 'orchestrating',
                      message: 'Processing notebook through orchestrator agents...',
                      progress: 85,
                    };
                    setCurrentUpdate(orchestratorUpdate);
                    
                    // Simulate orchestrator completion
                    setTimeout(() => {
                      const finalUpdate: StreamingUpdate = {
                        status: 'finished',
                        message: 'Notebook processing complete! Ready for report generation.',
                        progress: 100,
                        orchestrator_result: {
                          methodology_extracted: true,
                          results_analyzed: true,
                          figures_processed: true,
                          ready_for_report: true
                        }
                      };
                      setCurrentUpdate(finalUpdate);
                      onComplete({
                        cells: data.cells,
                        summary: data.summary,
                        orchestrator_result: finalUpdate.orchestrator_result
                      });
                    }, 2000);
                  }, 1000);
                }

                if (data.status === 'error') {
                  onError(data.error || 'Unknown error occurred');
                }
              } catch (e) {
                console.error('Error parsing SSE data:', e);
              }
            }
          }
        }
      } catch (error) {
        console.error('Streaming error:', error);
        onError(error instanceof Error ? error.message : 'Unknown error');
      }
    };

    processNotebook();
  }, [file, onComplete, onError]);

  if (!currentUpdate) {
    return (
      <Card className="p-8 bg-card border-border">
        <div className="flex items-center justify-center">
          <Loader2 className="w-8 h-8 animate-spin text-primary mr-3" />
          <span className="text-lg">Initializing notebook processor...</span>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Progress Header */}
      <Card className="p-6 bg-card border-border">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xl font-semibold text-foreground">
              {currentUpdate.status === 'starting' && 'Starting Analysis'}
              {currentUpdate.status === 'parsing' && 'Parsing Notebook'}
              {currentUpdate.status === 'completed' && 'Parsing Complete'}
              {currentUpdate.status === 'orchestrating' && 'Processing with AI Agents'}
              {currentUpdate.status === 'finished' && 'Analysis Complete'}
              {currentUpdate.status === 'error' && 'Error Occurred'}
            </h3>
            
            <div className="flex items-center gap-2">
              {currentUpdate.status === 'finished' ? (
                <CheckCircle className="w-5 h-5 text-green-400" />
              ) : (
                <Loader2 className="w-5 h-5 animate-spin text-primary" />
              )}
              <span className="text-sm text-muted-foreground">
                {currentUpdate.progress}%
              </span>
            </div>
          </div>
          
          <Progress value={currentUpdate.progress} className="h-2" />
          
          <p className="text-sm text-muted-foreground">
            {currentUpdate.message}
          </p>

          {/* Phase Indicators */}
          <div className="flex items-center gap-4 text-sm">
            <div className={`flex items-center gap-2 ${currentUpdate.progress > 0 ? 'text-green-400' : 'text-gray-500'}`}>
              <CheckCircle className={`w-4 h-4 ${currentUpdate.progress > 0 ? 'text-green-400' : 'text-gray-500'}`} />
              Notebook Parsing
            </div>
            <ArrowRight className="w-4 h-4 text-gray-500" />
            <div className={`flex items-center gap-2 ${isOrchestratorPhase ? 'text-blue-400' : 'text-gray-500'}`}>
              {isOrchestratorPhase ? (
                <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
              ) : (
                <div className="w-4 h-4 rounded-full border-2 border-gray-500" />
              )}
              AI Agent Processing
            </div>
            <ArrowRight className="w-4 h-4 text-gray-500" />
            <div className={`flex items-center gap-2 ${currentUpdate.status === 'finished' ? 'text-green-400' : 'text-gray-500'}`}>
              {currentUpdate.status === 'finished' ? (
                <CheckCircle className="w-4 h-4 text-green-400" />
              ) : (
                <div className="w-4 h-4 rounded-full border-2 border-gray-500" />
              )}
              Report Generation Ready
            </div>
          </div>
        </div>
      </Card>

      {/* Processed Cells Display */}
      {processedCells.length > 0 && (
        <Card className="p-6 bg-card border-border">
          <h4 className="text-lg font-semibold mb-4 text-foreground">
            Processed Cells ({processedCells.length})
          </h4>
          
          <Accordion type="single" collapsible className="space-y-2">
            {processedCells.map((cell, index) => (
              <AccordionItem 
                key={index} 
                value={`cell-${index}`} 
                className="border border-border rounded-lg animate-in slide-in-from-bottom-2 duration-300"
                style={{ animationDelay: `${index * 100}ms` }}
              >
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
        </Card>
      )}
    </div>
  );
};