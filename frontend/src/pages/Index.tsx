import { useState } from "react";
import axios from "axios";
import { FileUpload } from "@/components/FileUpload";
import { LoadingState } from "@/components/LoadingState";
import { ReportView } from "@/components/ReportView";
import { NotebookView } from "@/components/NotebookView";
import { PapergenWorkflow } from "@/components/PapergenWorkflow";
import { useToast } from "@/hooks/use-toast";
import { Sparkles, Brain } from "lucide-react";

interface ReportData {
  paper: string;
  analysis: string;
}

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

interface NotebookData {
  success: boolean;
  message: string;
  cells: NotebookCell[];
  summary: NotebookSummary;
  notebook_metadata?: any;
  orchestrator_result?: any;
}

const Index = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [reportData, setReportData] = useState<ReportData | null>(null);
  const [notebookData, setNotebookData] = useState<NotebookData | null>(null);
  const [papergenFile, setPapergenFile] = useState<File | null>(null);
  const [papergenResult, setPapergenResult] = useState<any>(null);
  const { toast } = useToast();

  const handleFileUpload = async (file: File) => {
    // Always use the enhanced workflow that shows complete process
    setIsLoading(false);
    setReportData(null);
    setNotebookData(null);
    setPapergenResult(null);
    setPapergenFile(file);
  };

  const handleStreamingComplete = (result: any) => {
    console.log("Streaming complete:", result);
    setPapergenFile(null);
    
    // Set notebook data for potential further processing
    setNotebookData({
      success: true,
      message: result.orchestrator_result ? "Analysis complete with AI processing" : "Notebook parsed successfully",
      cells: result.cells,
      summary: result.summary,
      orchestrator_result: result.orchestrator_result
    });

    toast({
      title: "Analysis Complete",
      description: "Your notebook has been successfully processed through the AI pipeline.",
    });
  };

  const handleStreamingError = (error: string) => {
    console.error("Streaming error:", error);
    setPapergenFile(null);
    
    toast({
      title: "Analysis Failed",
      description: error || "Failed to analyze notebook. Please try again.",
      variant: "destructive",
    });
  };

  const handlePapergenComplete = (result: any) => {
    console.log("Papergen complete:", result);
    setPapergenFile(null);
    setPapergenResult(result);
    
    toast({
      title: "Paper Generation Complete",
      description: "Your research paper has been successfully generated!",
    });
  };

  const handlePapergenError = (error: string) => {
    console.error("Papergen error:", error);
    setPapergenFile(null);
    
    toast({
      title: "Paper Generation Failed",
      description: error || "Failed to generate paper. Please try again.",
      variant: "destructive",
    });
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-12 max-w-6xl">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Sparkles className="w-10 h-10 text-primary" />
            <h1 className="text-5xl font-bold bg-gradient-primary bg-clip-text text-transparent">
              Praxis AI
            </h1>
          </div>
          <p className="text-xl text-muted-foreground">
            R&D Acceleration Platform
          </p>
          <p className="text-sm text-muted-foreground mt-2">
            Transform your Jupyter notebooks into comprehensive research papers and industry analysis
          </p>
        </div>

        {/* Main Content */}
        <div className="space-y-8">
          {!isLoading && !reportData && !notebookData && !papergenFile && !papergenResult && (
            <div className="space-y-6">
              <div className="text-center">
                <h2 className="text-3xl font-bold mb-4 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Complete AI Research Pipeline
                </h2>
                <p className="text-lg text-muted-foreground max-w-3xl mx-auto mb-4">
                  Upload your Jupyter notebook and watch our specialized AI agents collaborate in real-time to transform your code into a comprehensive, publication-ready research paper.
                </p>
                <div className="flex items-center justify-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                  <div className="flex items-center gap-1">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    <span>Live Progress Tracking</span>
                  </div>
                  <span>•</span>
                  <div className="flex items-center gap-1">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                    <span>Multi-Agent Coordination</span>
                  </div>
                  <span>•</span>
                  <div className="flex items-center gap-1">
                    <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse"></div>
                    <span>Professional LaTeX Output</span>
                  </div>
                </div>
              </div>
              
              <div className="bg-gradient-to-r from-blue-50 to-purple-50 p-6 rounded-xl border">
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <Brain className="h-5 w-5 text-purple-600" />
                  What happens when you upload?
                </h3>
                <div className="grid md:grid-cols-2 gap-4 text-sm text-gray-700 dark:text-gray-300">
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                      <span className="text-gray-800 dark:text-gray-200"><strong className="text-blue-700 dark:text-blue-300">Notebook Parser:</strong> Extracts code, imports, and outputs</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                      <span className="text-gray-800 dark:text-gray-200"><strong className="text-green-700 dark:text-green-300">Methodology Writer:</strong> Analyzes your technical approach</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                      <span className="text-gray-800 dark:text-gray-200"><strong className="text-orange-700 dark:text-orange-300">Results Analyzer:</strong> Interprets outputs and metrics</span>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-pink-500 rounded-full"></div>
                      <span className="text-gray-800 dark:text-gray-200"><strong className="text-pink-700 dark:text-pink-300">Literary Writer:</strong> Crafts academic introduction and abstract</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-indigo-500 rounded-full"></div>
                      <span className="text-gray-800 dark:text-gray-200"><strong className="text-indigo-700 dark:text-indigo-300">Illustration Generator:</strong> Creates technical diagrams</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-gray-500 rounded-full"></div>
                      <span className="text-gray-800 dark:text-gray-200"><strong className="text-gray-700 dark:text-gray-300">Document Formatter:</strong> Generates LaTeX and PDF</span>
                    </div>
                  </div>
                </div>
              </div>
              
              <FileUpload onFileSelect={handleFileUpload} isLoading={isLoading} />
            </div>
          )}

          {isLoading && <LoadingState />}

          {papergenFile && (
            <PapergenWorkflow
              file={papergenFile}
              onComplete={handlePapergenComplete}
              onError={handlePapergenError}
            />
          )}

          {papergenResult && (
            <>
              <div className="bg-green-50 border border-green-200 rounded-lg p-6 mb-8">
                <h2 className="text-xl font-bold text-green-800 mb-4">
                  ✅ Research Paper Generated Successfully!
                </h2>
                
                {papergenResult.paper_content && (
                  <div className="mb-4">
                    <h3 className="font-semibold mb-2">Generated Paper:</h3>
                    <div className="bg-white border rounded-lg p-4 max-h-96 overflow-y-auto">
                      <div className="text-gray-800 text-sm leading-relaxed">
                        {papergenResult.paper_content.split('\n').map((line, index) => (
                          <div key={index} className="mb-2">
                            {line.startsWith('#') ? (
                              <h4 className="font-bold text-lg text-gray-900 mt-4 mb-2">
                                {line.replace(/^#+\s*/, '')}
                              </h4>
                            ) : line.startsWith('##') ? (
                              <h5 className="font-semibold text-base text-gray-800 mt-3 mb-2">
                                {line.replace(/^#+\s*/, '')}
                              </h5>
                            ) : (
                              <p className="text-gray-700">{line}</p>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
                
                {papergenResult.sections && Object.keys(papergenResult.sections).length > 0 && (
                  <div className="mb-4">
                    <h3 className="font-semibold mb-2">Generated Sections:</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      {Object.keys(papergenResult.sections).map(section => (
                        <div key={section} className="bg-white border rounded px-2 py-1 text-sm text-gray-700 font-medium">
                          {section}
                        </div>
                      ))}
                    </div>
                    
                    {/* Show section contents */}
                    <div className="mt-4 space-y-4">
                      {Object.entries(papergenResult.sections).map(([sectionName, sectionContent]) => (
                        <div key={sectionName} className="bg-gray-50 border rounded-lg p-4">
                          <h4 className="font-semibold text-gray-800 mb-2 capitalize">
                            {sectionName.replace('_', ' ')}
                          </h4>
                          <div className="text-gray-700 text-sm leading-relaxed whitespace-pre-wrap">
                            {String(sectionContent || '')}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {papergenResult.illustrations && papergenResult.illustrations.length > 0 && (
                  <div className="mb-4">
                    <h3 className="font-semibold mb-2">
                      Illustrations Generated: {papergenResult.illustrations.length}
                    </h3>
                  </div>
                )}
              </div>
              
              <div className="flex justify-center">
                <button
                  onClick={() => {
                    setPapergenResult(null);
                    setNotebookData(null);
                    setReportData(null);
                  }}
                  className="text-sm text-muted-foreground hover:text-primary transition-colors underline"
                >
                  Generate another paper
                </button>
              </div>
            </>
          )}

          {notebookData && !isLoading && (
            <>
              <NotebookView 
                cells={notebookData.cells} 
                summary={notebookData.summary} 
                message={notebookData.message} 
              />
              
              {notebookData.orchestrator_result && (
                <div className="bg-green-900/20 p-4 rounded-lg border border-green-500/30">
                  <h4 className="text-lg font-semibold text-green-300 mb-2">
                    🤖 AI Processing Complete
                  </h4>
                  <p className="text-green-200 mb-3">
                    Your notebook has been processed by our orchestrator agents and is ready for report generation.
                  </p>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                    <div className="text-green-300">✅ Methodology Extracted</div>
                    <div className="text-green-300">✅ Results Analyzed</div>
                    <div className="text-green-300">✅ Figures Processed</div>
                    <div className="text-green-300">✅ Ready for Report</div>
                  </div>
                </div>
              )}
              
              <div className="flex justify-center">
                <button
                  onClick={() => {
                    setNotebookData(null);
                    setReportData(null);
                  }}
                  className="text-sm text-muted-foreground hover:text-primary transition-colors underline"
                >
                  Analyze another notebook
                </button>
              </div>
            </>
          )}

          {reportData && !isLoading && (
            <>
              <ReportView paper={reportData.paper} analysis={reportData.analysis} />
              <div className="flex justify-center">
                <button
                  onClick={() => {
                    setReportData(null);
                    setNotebookData(null);
                  }}
                  className="text-sm text-muted-foreground hover:text-primary transition-colors underline"
                >
                  Analyze another notebook
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default Index;
