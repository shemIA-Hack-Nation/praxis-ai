import { useState } from "react";
import axios from "axios";
import { FileUpload } from "@/components/FileUpload";
import { LoadingState } from "@/components/LoadingState";
import { ReportView } from "@/components/ReportView";
import { NotebookView } from "@/components/NotebookView";
import { StreamingNotebook } from "@/components/StreamingNotebook";
import { useToast } from "@/hooks/use-toast";
import { Sparkles } from "lucide-react";

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
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingFile, setStreamingFile] = useState<File | null>(null);
  const { toast } = useToast();

  const handleFileUpload = async (file: File) => {
    // Start streaming process
    setIsLoading(false);
    setReportData(null);
    setNotebookData(null);
    setIsStreaming(true);
    setStreamingFile(file);
  };

  const handleStreamingComplete = (result: any) => {
    console.log("Streaming complete:", result);
    setIsStreaming(false);
    setStreamingFile(null);
    
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
    setIsStreaming(false);
    setStreamingFile(null);
    
    toast({
      title: "Analysis Failed",
      description: error || "Failed to analyze notebook. Please try again.",
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
          {!isLoading && !reportData && !notebookData && !isStreaming && (
            <FileUpload onFileSelect={handleFileUpload} isLoading={isLoading || isStreaming} />
          )}

          {isLoading && <LoadingState />}

          {isStreaming && streamingFile && (
            <StreamingNotebook 
              file={streamingFile}
              onComplete={handleStreamingComplete}
              onError={handleStreamingError}
            />
          )}

          {notebookData && !isLoading && !isStreaming && (
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

          {reportData && !isLoading && !isStreaming && (
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
