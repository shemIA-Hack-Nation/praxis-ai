import { useState } from "react";
import axios from "axios";
import { FileUpload } from "@/components/FileUpload";
import { LoadingState } from "@/components/LoadingState";
import { ReportView } from "@/components/ReportView";
import { useToast } from "@/hooks/use-toast";
import { Sparkles } from "lucide-react";

interface ReportData {
  paper: string;
  analysis: string;
}

const Index = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [reportData, setReportData] = useState<ReportData | null>(null);
  const { toast } = useToast();

  const handleFileUpload = async (file: File) => {
    setIsLoading(true);
    setReportData(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("/api/v1/upload_notebook/", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setReportData(response.data);
      toast({
        title: "Analysis Complete",
        description: "Your notebook has been successfully analyzed.",
      });
    } catch (error) {
      console.error("Upload error:", error);
      toast({
        title: "Analysis Failed",
        description: axios.isAxiosError(error) 
          ? error.response?.data?.message || "Failed to analyze notebook. Please try again."
          : "An unexpected error occurred.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
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
          {!isLoading && !reportData && (
            <FileUpload onFileSelect={handleFileUpload} isLoading={isLoading} />
          )}

          {isLoading && <LoadingState />}

          {reportData && !isLoading && (
            <>
              <ReportView paper={reportData.paper} analysis={reportData.analysis} />
              <div className="flex justify-center">
                <button
                  onClick={() => setReportData(null)}
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
