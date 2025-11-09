import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';
import { CheckCircle2, Circle, XCircle, AlertCircle, Loader2, Brain, FileText, BarChart3, PenTool, Image, Sparkles } from 'lucide-react';

interface WorkflowMessage {
  phase: string;
  agent: string;
  stage?: string;
  status: 'starting' | 'working' | 'completed' | 'error';
  message: string;
  progress: number;
  data?: any;
}

interface PapergenWorkflowProps {
  file: File;
  onComplete: (result: any) => void;
  onError: (error: string) => void;
}

const AGENT_ICONS = {
  'NotebookParser': FileText,
  'PapergenOrchestrator': Brain,
  'MethodologyWriter': PenTool,
  'ResultsWriter': BarChart3,
  'LiteraryAgent': Sparkles,
  'IllustrationCritic': Image,
  'FormatterAgent': FileText,
  'System': AlertCircle
};

const AGENT_COLORS = {
  'NotebookParser': 'bg-blue-100 text-blue-800',
  'PapergenOrchestrator': 'bg-purple-100 text-purple-800',
  'MethodologyWriter': 'bg-green-100 text-green-800',
  'ResultsWriter': 'bg-orange-100 text-orange-800',
  'LiteraryAgent': 'bg-pink-100 text-pink-800',
  'IllustrationCritic': 'bg-indigo-100 text-indigo-800',
  'FormatterAgent': 'bg-gray-100 text-gray-800',
  'System': 'bg-red-100 text-red-800'
};

const STATUS_ICONS = {
  'starting': Loader2,
  'working': Loader2,
  'completed': CheckCircle2,
  'error': XCircle
};

export const PapergenWorkflow: React.FC<PapergenWorkflowProps> = ({
  file,
  onComplete,
  onError
}) => {
  const [messages, setMessages] = useState<WorkflowMessage[]>([]);
  const [currentProgress, setCurrentProgress] = useState(0);
  const [isStreaming, setIsStreaming] = useState(false);
  const [finalResult, setFinalResult] = useState<any>(null);

  useEffect(() => {
    if (file) {
      startPaperGeneration();
    }
  }, [file]);

  const startPaperGeneration = async () => {
    setIsStreaming(true);
    setMessages([]);
    setCurrentProgress(0);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/api/v1/generate_paper_stream/', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body reader available');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              const message: WorkflowMessage = {
                phase: data.phase,
                agent: data.agent,
                stage: data.stage,
                status: data.status,
                message: data.message,
                progress: data.progress,
                data: data.data
              };
              
              setMessages(prev => [...prev, message]);
              setCurrentProgress(data.progress);
              
              // Check if this is the final result
              if (data.phase === 'completion' && data.data) {
                setFinalResult(data.data);
                if (data.status === 'completed') {
                  onComplete(data.data);
                } else if (data.status === 'error') {
                  onError(data.message);
                }
              }
            } catch (e) {
              console.error('Failed to parse SSE data:', e);
            }
          }
        }
      }
    } catch (error) {
      console.error('Error during paper generation:', error);
      onError(error instanceof Error ? error.message : 'Unknown error occurred');
    } finally {
      setIsStreaming(false);
    }
  };

  const getStatusIcon = (status: string) => {
    const IconComponent = STATUS_ICONS[status as keyof typeof STATUS_ICONS] || Circle;
    return IconComponent;
  };

  const getAgentIcon = (agent: string) => {
    const IconComponent = AGENT_ICONS[agent as keyof typeof AGENT_ICONS] || Circle;
    return IconComponent;
  };

  const getAgentColor = (agent: string) => {
    return AGENT_COLORS[agent as keyof typeof AGENT_COLORS] || 'bg-gray-100 text-gray-800';
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-600 dark:text-green-400';
      case 'error': return 'text-red-600 dark:text-red-400';
      case 'starting':
      case 'working': return 'text-blue-600 dark:text-blue-400';
      default: return 'text-gray-600 dark:text-gray-400';
    }
  };

  const getAgentDescription = (agent: string) => {
    const descriptions = {
      'NotebookParser': 'Analyzes your Jupyter notebook structure, extracts code patterns, imports, and execution outputs',
      'PapergenOrchestrator': 'Coordinates the multi-agent workflow and manages the overall paper generation process',
      'MethodologyWriter': 'Examines your technical approach, algorithms, and experimental design to craft the methodology section',
      'ResultsWriter': 'Interprets your output data, metrics, and visualizations to create comprehensive results analysis',
      'LiteraryAgent': 'Crafts academic introduction, abstract, and conclusion with proper scholarly language and structure',
      'IllustrationCritic': 'Generates technical diagrams, charts, and visualizations to enhance paper comprehension',
      'FormatterAgent': 'Assembles all sections into professional LaTeX format and compiles the final research paper',
      'System': 'Handles system operations and error management'
    };
    return descriptions[agent as keyof typeof descriptions] || 'Specialized AI agent processing your research content';
  };

  const getWorkflowSteps = () => [
    {
      name: 'Notebook Parser',
      icon: FileText,
      description: 'Analyzes your Jupyter notebook structure, extracts code patterns, imports, and execution outputs'
    },
    {
      name: 'Methodology Writer',
      icon: PenTool,
      description: 'Examines your technical approach, algorithms, and experimental design to craft the methodology section'
    },
    {
      name: 'Results Writer',
      icon: BarChart3,
      description: 'Interprets your output data, metrics, and visualizations to create comprehensive results analysis'
    },
    {
      name: 'Literary Agent',
      icon: Sparkles,
      description: 'Crafts academic introduction, abstract, and conclusion with proper scholarly language and structure'
    },
    {
      name: 'Illustration Generator',
      icon: Image,
      description: 'Generates technical diagrams, charts, and visualizations to enhance paper comprehension'
    },
    {
      name: 'Document Formatter',
      icon: FileText,
      description: 'Assembles all sections into professional LaTeX format and compiles the final research paper'
    }
  ];

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="h-5 w-5" />
          Research Paper Generation Workflow
        </CardTitle>
        <Progress value={currentProgress} className="w-full" />
        <div className="text-sm text-gray-600">
          Progress: {currentProgress}%
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Multi-Agent Workflow Overview */}
        <div className="mb-6 p-4 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg border">
          <h3 className="text-lg font-semibold mb-3 text-gray-800 dark:text-gray-200 flex items-center gap-2">
            <Brain className="h-5 w-5 text-purple-600" />
            AI Multi-Agent Research Pipeline
          </h3>
          <div className="text-sm text-gray-600 dark:text-gray-400 mb-4">
            Watch as our specialized AI agents collaborate to transform your Jupyter notebook into a comprehensive research paper:
          </div>
          
          {/* Agent Status Grid */}
          {messages.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {Array.from(new Set(messages.map(m => m.agent))).map(agent => {
                const agentMessages = messages.filter(m => m.agent === agent);
                const lastStatus = agentMessages[agentMessages.length - 1]?.status || 'starting';
                const AgentIcon = getAgentIcon(agent);
                const StatusIcon = getStatusIcon(lastStatus);
                const agentDescription = getAgentDescription(agent);
                
                return (
                  <div key={agent} className={`p-3 rounded-lg border transition-all duration-300 ${
                    lastStatus === 'completed' ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800' :
                    lastStatus === 'error' ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800' :
                    lastStatus === 'working' ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800' :
                    'bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-700'
                  }`}>
                    <div className="flex items-center gap-2 mb-2">
                      <AgentIcon className="h-5 w-5 text-gray-700 dark:text-gray-300" />
                      <span className="font-medium text-gray-800 dark:text-gray-200">{agent}</span>
                      <StatusIcon className={`h-4 w-4 ml-auto ${getStatusColor(lastStatus)} ${
                        lastStatus === 'working' || lastStatus === 'starting' ? 'animate-spin' : ''
                      }`} />
                    </div>
                    <div className="text-xs text-gray-600 dark:text-gray-400 mb-2">
                      {agentDescription}
                    </div>
                    <div className={`text-xs font-medium ${getStatusColor(lastStatus)}`}>
                      {lastStatus === 'starting' ? '🔄 Initializing...' : 
                       lastStatus === 'working' ? '⚡ Processing...' :
                       lastStatus === 'completed' ? '✅ Completed' : 
                       lastStatus === 'error' ? '❌ Failed' : '⏳ Waiting...'}
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {getWorkflowSteps().map((step, index) => (
                <div key={index} className="p-3 rounded-lg bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700">
                  <div className="flex items-center gap-2 mb-2">
                    <step.icon className="h-5 w-5 text-gray-600 dark:text-gray-400" />
                    <span className="font-medium text-gray-800 dark:text-gray-200">{step.name}</span>
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">
                    {step.description}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Real-time Workflow Progress */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Live Workflow Progress
          </h3>
          
          <div className="space-y-3 max-h-96 overflow-y-auto bg-gray-50 dark:bg-gray-900 p-4 rounded-lg border">
            {messages.length === 0 && (
              <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                <Loader2 className="h-8 w-8 animate-spin mx-auto mb-2" />
                <div>Waiting for workflow to begin...</div>
                <div className="text-sm">Upload your notebook to start the AI research paper generation</div>
              </div>
            )}
            
            {messages.map((message, index) => {
              const AgentIcon = getAgentIcon(message.agent);
              const StatusIcon = getStatusIcon(message.status);
              
              return (
                <div 
                  key={index}
                  className={`flex items-start gap-3 p-4 rounded-lg border transition-all duration-300 shadow-sm ${
                    message.status === 'error' 
                      ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800' 
                      : message.status === 'completed'
                      ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
                      : message.status === 'working' || message.status === 'starting'
                      ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800'
                      : 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700'
                  }`}
                >
                  <div className="flex items-center gap-2 min-w-0 flex-shrink-0">
                    <AgentIcon className="h-5 w-5 text-gray-700 dark:text-gray-300" />
                    <Badge variant="secondary" className={`${getAgentColor(message.agent)} text-xs px-2 py-1`}>
                      {message.agent}
                    </Badge>
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-2">
                      <StatusIcon 
                        className={`h-4 w-4 ${getStatusColor(message.status)} ${
                          message.status === 'working' || message.status === 'starting' ? 'animate-spin' : ''
                        }`} 
                      />
                      {message.stage && (
                        <span className="text-sm font-medium text-gray-800 dark:text-gray-200">
                          {message.stage}
                        </span>
                      )}
                      <span className="text-xs text-gray-500 dark:text-gray-400 ml-auto">
                        Progress: {message.progress}%
                      </span>
                    </div>
                    
                    <div className="text-sm text-gray-800 dark:text-gray-200 mb-2">
                      {message.message}
                    </div>
                    
                    {/* Enhanced output display for agent results */}
                    {message.data && (
                      <div className="mt-3 p-3 bg-white dark:bg-gray-800 rounded border">
                        <div className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-2">
                          🔍 Agent Output Details:
                        </div>
                        
                        {message.data.cell_count && (
                          <div className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400 mb-1">
                            <FileText className="h-3 w-3" />
                            <span>Processed {message.data.cell_count} notebook cells</span>
                          </div>
                        )}
                        
                        {message.data.agents_status && (
                          <div className="mb-2">
                            <div className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">Agent Status:</div>
                            <div className="flex flex-wrap gap-1">
                              {Object.entries(message.data.agents_status).map(([agent, available]) => (
                                <Badge 
                                  key={agent} 
                                  variant={available ? "default" : "secondary"}
                                  className="text-xs"
                                >
                                  {agent}: {available ? '✅' : '❌'}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        )}
                        
                        {message.data.sections && (
                          <div className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400 mb-1">
                            <PenTool className="h-3 w-3" />
                            <span>Generated sections: {Object.keys(message.data.sections).join(', ')}</span>
                          </div>
                        )}
                        
                        {message.data.illustrations && Array.isArray(message.data.illustrations) && (
                          <div className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400 mb-1">
                            <Image className="h-3 w-3" />
                            <span>Created {message.data.illustrations.length} illustrations</span>
                          </div>
                        )}
                        
                        {message.data.paper_content && (
                          <div className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400 mb-1">
                            <FileText className="h-3 w-3" />
                            <span>Paper length: {message.data.paper_content.length} characters</span>
                          </div>
                        )}
                        
                        {message.data.latex_file && (
                          <div className="flex items-center gap-2 text-xs text-green-600 dark:text-green-400 mb-1">
                            <CheckCircle2 className="h-3 w-3" />
                            <span>LaTeX file generated successfully</span>
                          </div>
                        )}
                        
                        {message.data.pdf_file && (
                          <div className="flex items-center gap-2 text-xs text-green-600 dark:text-green-400 mb-1">
                            <CheckCircle2 className="h-3 w-3" />
                            <span>PDF compiled successfully</span>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Loading indicator */}
        {isStreaming && (
          <div className="flex items-center justify-center gap-2 p-4 text-blue-600">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>Processing workflow...</span>
          </div>
        )}

        {/* Research Paper Generation Results */}
        {finalResult && (
          <div className="mt-6 p-6 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 border border-green-200 dark:border-green-800 rounded-lg">
            <div className="flex items-center gap-3 mb-4">
              <div className="flex-shrink-0">
                <CheckCircle2 className="h-8 w-8 text-green-600 dark:text-green-400" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-green-800 dark:text-green-200">
                  🎉 Research Paper Generated Successfully!
                </h3>
                <p className="text-sm text-green-700 dark:text-green-300">
                  Your Jupyter notebook has been transformed into a comprehensive academic research paper
                </p>
              </div>
            </div>
            
            {/* Paper Statistics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              {finalResult.sections && (
                <div className="bg-white dark:bg-gray-800 p-3 rounded-lg border">
                  <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                    {Object.keys(finalResult.sections).length}
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Sections Generated</div>
                  <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                    {Object.keys(finalResult.sections).join(', ')}
                  </div>
                </div>
              )}
              
              {finalResult.illustrations && (
                <div className="bg-white dark:bg-gray-800 p-3 rounded-lg border">
                  <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                    {finalResult.illustrations.length}
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Illustrations</div>
                  <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                    Diagrams & Charts
                  </div>
                </div>
              )}
              
              {finalResult.paper_content && (
                <div className="bg-white dark:bg-gray-800 p-3 rounded-lg border">
                  <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                    {Math.round(finalResult.paper_content.length / 1000)}k
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Characters</div>
                  <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                    ~{Math.round(finalResult.paper_content.split(' ').length)} words
                  </div>
                </div>
              )}
              
              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg border">
                <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                  {finalResult.latex_url ? 'LaTeX' : 'HTML'}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Format</div>
                <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                  Professional Quality
                </div>
              </div>
            </div>
            
            {/* Paper Preview */}
            {finalResult.sections && (
              <div className="mb-6">
                <h4 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-3">
                  📋 Generated Paper Structure
                </h4>
                <div className="grid gap-2">
                  {Object.entries(finalResult.sections).map(([section, content]) => (
                    <div key={section} className="bg-white dark:bg-gray-800 p-3 rounded border">
                      <div className="flex items-center justify-between">
                        <span className="font-medium capitalize text-gray-800 dark:text-gray-200">
                          {section.replace('_', ' ')}
                        </span>
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          {typeof content === 'string' ? `${content.length} chars` : 'Generated'}
                        </span>
                      </div>
                      {typeof content === 'string' && content && (
                        <div className="text-xs text-gray-600 dark:text-gray-400 mt-1 truncate">
                          {content.substring(0, 100)}...
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {/* Download Options */}
            <div className="flex flex-col sm:flex-row gap-3">
              {finalResult.pdf_url && (
                <a
                  href={`http://localhost:8000${finalResult.pdf_url}`}
                  download
                  className="inline-flex items-center justify-center px-6 py-3 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 text-white font-medium rounded-lg transition-all duration-200 shadow-lg hover:shadow-xl"
                >
                  <FileText className="h-5 w-5 mr-2" />
                  Download PDF Research Paper
                  <span className="ml-2 text-xs bg-red-800 px-2 py-1 rounded">Recommended</span>
                </a>
              )}
              
              {finalResult.latex_url && (
                <a
                  href={`http://localhost:8000${finalResult.latex_url}`}
                  download
                  className="inline-flex items-center justify-center px-6 py-3 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-medium rounded-lg transition-all duration-200 shadow-lg hover:shadow-xl"
                >
                  <FileText className="h-5 w-5 mr-2" />
                  Download LaTeX Source
                  <span className="ml-2 text-xs bg-blue-800 px-2 py-1 rounded">Advanced</span>
                </a>
              )}
              
              {!finalResult.pdf_url && !finalResult.latex_url && (
                <div className="text-sm text-gray-600 dark:text-gray-400 italic p-3 bg-gray-100 dark:bg-gray-700 rounded">
                  ⚠️ Download files are being prepared. Please check back in a moment.
                </div>
              )}
            </div>
            
            {/* Next Steps */}
            <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
              <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">
                🚀 What's Next?
              </h4>
              <ul className="text-sm text-blue-700 dark:text-blue-300 space-y-1">
                <li>• Review the generated research paper for accuracy and completeness</li>
                <li>• Edit and customize the content to match your specific requirements</li>
                <li>• Add citations and references as needed for publication</li>
                <li>• Use the LaTeX source for journal submissions or further formatting</li>
              </ul>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};