import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from './ui/collapsible';
import { 
  CheckCircle2, 
  Circle, 
  XCircle, 
  AlertCircle, 
  Loader2, 
  Brain, 
  FileText, 
  BarChart3, 
  PenTool, 
  Image, 
  Sparkles,
  ChevronDown,
  ChevronUp,
  Eye,
  Download,
  Code2,
  Cpu,
  Microscope,
  Palette,
  BookOpen,
  Zap
} from 'lucide-react';

interface WorkflowMessage {
  phase: string;
  agent: string;
  stage?: string;
  status: 'starting' | 'working' | 'completed' | 'error';
  message: string;
  progress: number;
  data?: any;
  timestamp?: string;
}

interface EnhancedPapergenWorkflowProps {
  file: File;
  onComplete: (result: any) => void;
  onError: (error: string) => void;
}

// Enhanced agent configuration with detailed descriptions
const AGENT_CONFIG = {
  'NotebookParser': {
    icon: Code2,
    name: 'Notebook Parser',
    description: 'Extracts and analyzes code structure, imports, and outputs from Jupyter notebooks',
    color: 'bg-blue-100 text-blue-800 border-blue-200',
    phases: ['notebook_parsing'],
    outputs: ['Code cells', 'Markdown cells', 'Outputs', 'Dependencies']
  },
  'PapergenOrchestrator': {
    icon: Brain,
    name: 'AI Orchestrator',
    description: 'Coordinates all AI agents and manages the research paper generation workflow',
    color: 'bg-purple-100 text-purple-800 border-purple-200',
    phases: ['orchestrator_init', 'workflow_start'],
    outputs: ['Workflow coordination', 'Agent management', 'Task distribution']
  },
  'MethodologyWriter': {
    icon: Microscope,
    name: 'Methodology Writer',
    description: 'Analyzes code structure and generates comprehensive methodology sections',
    color: 'bg-green-100 text-green-800 border-green-200',
    phases: ['methodology_writing'],
    outputs: ['Technical approach', 'Algorithm description', 'Implementation details']
  },
  'ResultsWriter': {
    icon: BarChart3,
    name: 'Results Analyzer',
    description: 'Interprets outputs, metrics, and generates results and analysis sections',
    color: 'bg-orange-100 text-orange-800 border-orange-200',
    phases: ['results_writing'],
    outputs: ['Performance metrics', 'Statistical analysis', 'Findings summary']
  },
  'LiteraryAgent': {
    icon: BookOpen,
    name: 'Literary Writer',
    description: 'Crafts academic abstracts, introductions, and conclusions with proper citations',
    color: 'bg-pink-100 text-pink-800 border-pink-200',
    phases: ['literary_writing'],
    outputs: ['Abstract', 'Introduction', 'Conclusion', 'References']
  },
  'IllustrationCritic': {
    icon: Palette,
    name: 'Illustration Generator',
    description: 'Creates technical diagrams, charts, and visualizations for the research paper',
    color: 'bg-indigo-100 text-indigo-800 border-indigo-200',
    phases: ['illustration_review'],
    outputs: ['Architecture diagrams', 'Performance charts', 'Technical illustrations']
  },
  'FormatterAgent': {
    icon: FileText,
    name: 'Document Formatter',
    description: 'Assembles final paper, generates LaTeX code, and compiles PDF documents',
    color: 'bg-gray-100 text-gray-800 border-gray-200',
    phases: ['final_assembly', 'completion'],
    outputs: ['Formatted paper', 'LaTeX source', 'PDF document']
  }
};

const WORKFLOW_PHASES = {
  'notebook_parsing': {
    title: 'Notebook Analysis',
    description: 'Parsing and understanding your Jupyter notebook',
    order: 1
  },
  'orchestrator_init': {
    title: 'AI System Initialization',
    description: 'Setting up the multi-agent research system',
    order: 2
  },
  'workflow_start': {
    title: 'Workflow Execution',
    description: 'Beginning the research paper generation process',
    order: 3
  },
  'methodology_writing': {
    title: 'Technical Analysis',
    description: 'Analyzing code structure and methodology',
    order: 4
  },
  'results_writing': {
    title: 'Results Processing',
    description: 'Interpreting outputs and generating findings',
    order: 5
  },
  'literary_writing': {
    title: 'Academic Writing',
    description: 'Creating scholarly introduction and abstract',
    order: 6
  },
  'illustration_review': {
    title: 'Visual Generation',
    description: 'Creating technical diagrams and charts',
    order: 7
  },
  'final_assembly': {
    title: 'Document Assembly',
    description: 'Compiling final research paper',
    order: 8
  },
  'completion': {
    title: 'Completion',
    description: 'Paper generation finished',
    order: 9
  }
};

const STATUS_ICONS = {
  'starting': Loader2,
  'working': Loader2,
  'completed': CheckCircle2,
  'error': XCircle
};

export const EnhancedPapergenWorkflow: React.FC<EnhancedPapergenWorkflowProps> = ({
  file,
  onComplete,
  onError
}) => {
  const [messages, setMessages] = useState<WorkflowMessage[]>([]);
  const [currentProgress, setCurrentProgress] = useState(0);
  const [isStreaming, setIsStreaming] = useState(false);
  const [finalResult, setFinalResult] = useState<any>(null);
  const [expandedAgents, setExpandedAgents] = useState<Set<string>>(new Set());
  const [currentPhase, setCurrentPhase] = useState<string>('');

  useEffect(() => {
    if (file) {
      startPaperGeneration();
    }
  }, [file]);

  const startPaperGeneration = async () => {
    setIsStreaming(true);
    setMessages([]);
    setCurrentProgress(0);
    setExpandedAgents(new Set());
    
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
                data: data.data,
                timestamp: new Date().toISOString()
              };
              
              setMessages(prev => [...prev, message]);
              setCurrentProgress(data.progress);
              setCurrentPhase(data.phase);
              
              // Auto-expand the current agent
              if (data.agent) {
                setExpandedAgents(prev => new Set([...prev, data.agent]));
              }
              
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

  const toggleAgentExpanded = (agentName: string) => {
    setExpandedAgents(prev => {
      const newSet = new Set(prev);
      if (newSet.has(agentName)) {
        newSet.delete(agentName);
      } else {
        newSet.add(agentName);
      }
      return newSet;
    });
  };

  const getAgentMessages = (agentName: string) => {
    return messages.filter(m => m.agent === agentName);
  };

  const getAgentStatus = (agentName: string) => {
    const agentMessages = getAgentMessages(agentName);
    if (agentMessages.length === 0) return 'waiting';
    return agentMessages[agentMessages.length - 1]?.status || 'waiting';
  };

  const getPhaseStatus = (phaseName: string) => {
    const phaseMessages = messages.filter(m => m.phase === phaseName);
    if (phaseMessages.length === 0) return 'waiting';
    
    const hasError = phaseMessages.some(m => m.status === 'error');
    const hasCompleted = phaseMessages.some(m => m.status === 'completed');
    
    if (hasError) return 'error';
    if (hasCompleted) return 'completed';
    return 'in-progress';
  };

  const formatTimestamp = (timestamp?: string) => {
    if (!timestamp) return '';
    return new Date(timestamp).toLocaleTimeString();
  };

  return (
    <div className="w-full max-w-6xl mx-auto space-y-6">
      {/* Header Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-3">
            <Zap className="h-6 w-6 text-purple-600" />
            <div>
              <div className="text-xl">AI-Powered Research Paper Generation</div>
              <div className="text-sm text-gray-600 font-normal mt-1">
                Transforming your Jupyter notebook into a comprehensive research paper
              </div>
            </div>
          </CardTitle>
          <div className="space-y-2">
            <Progress value={currentProgress} className="w-full h-3" />
            <div className="flex justify-between text-sm text-gray-600">
              <span>Progress: {currentProgress}%</span>
              <span className="capitalize">
                {currentPhase ? WORKFLOW_PHASES[currentPhase as keyof typeof WORKFLOW_PHASES]?.title || currentPhase : 'Initializing...'}
              </span>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Workflow Timeline */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Cpu className="h-5 w-5" />
            Workflow Timeline
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3">
            {Object.entries(WORKFLOW_PHASES).map(([phaseKey, phase]) => {
              const status = getPhaseStatus(phaseKey);
              const isActive = currentPhase === phaseKey;
              
              return (
                <div
                  key={phaseKey}
                  className={`flex items-center gap-4 p-3 rounded-lg border transition-all ${
                    isActive ? 'border-blue-500 bg-blue-50' :
                    status === 'completed' ? 'border-green-200 bg-green-50' :
                    status === 'error' ? 'border-red-200 bg-red-50' :
                    'border-gray-200 bg-gray-50'
                  }`}
                >
                  <div className="flex items-center justify-center w-8 h-8 rounded-full bg-white border-2">
                    {status === 'completed' ? (
                      <CheckCircle2 className="h-5 w-5 text-green-600" />
                    ) : status === 'error' ? (
                      <XCircle className="h-5 w-5 text-red-600" />
                    ) : isActive ? (
                      <Loader2 className="h-4 w-4 text-blue-600 animate-spin" />
                    ) : (
                      <div className="w-3 h-3 rounded-full bg-gray-300" />
                    )}
                  </div>
                  <div className="flex-1">
                    <div className="font-medium text-gray-900">{phase.title}</div>
                    <div className="text-sm text-gray-600">{phase.description}</div>
                  </div>
                  <Badge variant={
                    status === 'completed' ? 'default' :
                    status === 'error' ? 'destructive' :
                    isActive ? 'secondary' : 'outline'
                  }>
                    {status === 'waiting' ? 'Waiting' :
                     status === 'in-progress' ? 'In Progress' :
                     status === 'completed' ? 'Completed' : 'Error'}
                  </Badge>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* AI Agents Dashboard */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Brain className="h-5 w-5" />
            AI Agents Dashboard
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {Object.entries(AGENT_CONFIG).map(([agentKey, agent]) => {
            const agentMessages = getAgentMessages(agentKey);
            const status = getAgentStatus(agentKey);
            const isExpanded = expandedAgents.has(agentKey);
            const IconComponent = agent.icon;
            
            if (agentMessages.length === 0 && status === 'waiting') {
              return null; // Don't show agents that haven't been activated yet
            }
            
            return (
              <Collapsible key={agentKey} open={isExpanded}>
                <Card className={`border-l-4 ${agent.color.split(' ')[2]} transition-all duration-200`}>
                  <CollapsibleTrigger asChild>
                    <CardHeader 
                      className="cursor-pointer hover:bg-gray-50 transition-colors"
                      onClick={() => toggleAgentExpanded(agentKey)}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div className={`p-2 rounded-lg ${agent.color}`}>
                            <IconComponent className="h-5 w-5" />
                          </div>
                          <div className="text-left">
                            <CardTitle className="text-base">{agent.name}</CardTitle>
                            <p className="text-sm text-gray-600 mt-1">{agent.description}</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge variant={
                            status === 'completed' ? 'default' :
                            status === 'error' ? 'destructive' :
                            status === 'starting' || status === 'working' ? 'secondary' : 'outline'
                          }>
                            {status === 'starting' ? 'Starting' :
                             status === 'working' ? 'Working' :
                             status === 'completed' ? 'Completed' :
                             status === 'error' ? 'Error' : 'Waiting'}
                          </Badge>
                          {isExpanded ? (
                            <ChevronUp className="h-4 w-4" />
                          ) : (
                            <ChevronDown className="h-4 w-4" />
                          )}
                        </div>
                      </div>
                    </CardHeader>
                  </CollapsibleTrigger>
                  
                  <CollapsibleContent>
                    <CardContent className="pt-0">
                      {/* Agent Output Expectations */}
                      <div className="mb-4">
                        <h4 className="text-sm font-medium text-gray-700 mb-2">Expected Outputs:</h4>
                        <div className="flex flex-wrap gap-1">
                          {agent.outputs.map((output) => (
                            <Badge key={output} variant="outline" className="text-xs">
                              {output}
                            </Badge>
                          ))}
                        </div>
                      </div>
                      
                      {/* Agent Messages */}
                      <div className="space-y-2">
                        <h4 className="text-sm font-medium text-gray-700">Activity Log:</h4>
                        {agentMessages.length > 0 ? (
                          <div className="space-y-2 max-h-40 overflow-y-auto">
                            {agentMessages.map((message, index) => {
                              const StatusIcon = STATUS_ICONS[message.status as keyof typeof STATUS_ICONS];
                              
                              return (
                                <div
                                  key={index}
                                  className={`flex items-start gap-2 p-2 rounded text-sm ${
                                    message.status === 'error' ? 'bg-red-50 text-red-800' :
                                    message.status === 'completed' ? 'bg-green-50 text-green-800' :
                                    'bg-blue-50 text-blue-800'
                                  }`}
                                >
                                  <StatusIcon className={`h-4 w-4 mt-0.5 flex-shrink-0 ${
                                    message.status === 'working' || message.status === 'starting' ? 'animate-spin' : ''
                                  }`} />
                                  <div className="flex-1">
                                    <div className="flex items-center justify-between">
                                      <span>{message.message}</span>
                                      <span className="text-xs opacity-75">
                                        {formatTimestamp(message.timestamp)}
                                      </span>
                                    </div>
                                    {message.data && (
                                      <div className="mt-1 text-xs opacity-75">
                                        {message.data.cell_count && `Processed ${message.data.cell_count} cells`}
                                        {message.data.sections && `Generated sections: ${Object.keys(message.data.sections).join(', ')}`}
                                        {message.data.illustrations && `Created ${message.data.illustrations.length} illustrations`}
                                      </div>
                                    )}
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        ) : (
                          <div className="text-sm text-gray-500 italic p-2 bg-gray-50 rounded">
                            No activity recorded yet
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </CollapsibleContent>
                </Card>
              </Collapsible>
            );
          })}
        </CardContent>
      </Card>

      {/* Real-time Activity Feed */}
      {messages.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Eye className="h-5 w-5" />
              Real-time Activity Feed
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {messages.slice().reverse().slice(0, 10).map((message, index) => {
                const AgentIcon = AGENT_CONFIG[message.agent as keyof typeof AGENT_CONFIG]?.icon || Circle;
                const StatusIcon = STATUS_ICONS[message.status as keyof typeof STATUS_ICONS];
                
                return (
                  <div 
                    key={index}
                    className="flex items-center gap-3 p-2 rounded-lg bg-gray-50 border text-sm"
                  >
                    <AgentIcon className="h-4 w-4 text-gray-600" />
                    <StatusIcon className={`h-3 w-3 ${
                      message.status === 'error' ? 'text-red-600' :
                      message.status === 'completed' ? 'text-green-600' :
                      'text-blue-600'
                    } ${message.status === 'working' || message.status === 'starting' ? 'animate-spin' : ''}`} />
                    <div className="flex-1">
                      <span className="font-medium">{message.agent}:</span> {message.message}
                    </div>
                    <span className="text-xs text-gray-500">
                      {formatTimestamp(message.timestamp)}
                    </span>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Final Results */}
      {finalResult && (
        <Card className="border-green-200 bg-green-50">
          <CardHeader>
            <CardTitle className="text-lg text-green-800 flex items-center gap-2">
              <CheckCircle2 className="h-6 w-6" />
              Research Paper Generated Successfully!
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Paper Statistics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {finalResult.sections && (
                  <div className="text-center p-3 bg-white rounded-lg border">
                    <div className="text-2xl font-bold text-blue-600">
                      {Object.keys(finalResult.sections).length}
                    </div>
                    <div className="text-sm text-gray-600">Sections</div>
                  </div>
                )}
                {finalResult.illustrations && (
                  <div className="text-center p-3 bg-white rounded-lg border">
                    <div className="text-2xl font-bold text-purple-600">
                      {finalResult.illustrations.length}
                    </div>
                    <div className="text-sm text-gray-600">Illustrations</div>
                  </div>
                )}
                {finalResult.paper_content && (
                  <div className="text-center p-3 bg-white rounded-lg border">
                    <div className="text-2xl font-bold text-green-600">
                      {Math.round(finalResult.paper_content.length / 1000)}k
                    </div>
                    <div className="text-sm text-gray-600">Characters</div>
                  </div>
                )}
                <div className="text-center p-3 bg-white rounded-lg border">
                  <div className="text-2xl font-bold text-orange-600">
                    {messages.filter(m => m.status === 'completed').length}
                  </div>
                  <div className="text-sm text-gray-600">Tasks Completed</div>
                </div>
              </div>
              
              {/* Download Options */}
              <div className="flex flex-wrap gap-3 justify-center">
                {finalResult.pdf_url && (
                  <Button asChild className="bg-red-600 hover:bg-red-700">
                    <a href={`http://localhost:8000${finalResult.pdf_url}`} download>
                      <Download className="h-4 w-4 mr-2" />
                      Download PDF
                    </a>
                  </Button>
                )}
                {finalResult.latex_url && (
                  <Button asChild variant="outline">
                    <a href={`http://localhost:8000${finalResult.latex_url}`} download>
                      <Download className="h-4 w-4 mr-2" />
                      Download LaTeX Source
                    </a>
                  </Button>
                )}
                {finalResult.paper_content && (
                  <Button 
                    variant="outline"
                    onClick={() => {
                      const blob = new Blob([finalResult.paper_content], { type: 'text/markdown' });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = 'research_paper.md';
                      document.body.appendChild(a);
                      a.click();
                      document.body.removeChild(a);
                      URL.revokeObjectURL(url);
                    }}
                  >
                    <Download className="h-4 w-4 mr-2" />
                    Download Markdown
                  </Button>
                )}
              </div>

              {/* Generated Sections Preview */}
              {finalResult.sections && (
                <div className="mt-4">
                  <h4 className="font-medium text-green-800 mb-2">Generated Sections:</h4>
                  <div className="flex flex-wrap gap-2">
                    {Object.keys(finalResult.sections).map((section) => (
                      <Badge key={section} variant="secondary" className="bg-green-100 text-green-800">
                        {section.charAt(0).toUpperCase() + section.slice(1)}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Processing Indicator */}
      {isStreaming && (
        <Card className="border-blue-200 bg-blue-50">
          <CardContent className="flex items-center justify-center gap-3 py-8">
            <Loader2 className="h-6 w-6 animate-spin text-blue-600" />
            <div className="text-center">
              <div className="font-medium text-blue-800">AI System Processing...</div>
              <div className="text-sm text-blue-600 mt-1">
                {currentPhase ? WORKFLOW_PHASES[currentPhase as keyof typeof WORKFLOW_PHASES]?.title || 'Working...' : 'Initializing workflow...'}
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};