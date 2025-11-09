import { useState, useRef } from "react";
import axios from "axios";
import { FileUpload } from "@/components/FileUpload";
import { LoadingState } from "@/components/LoadingState";
import { ReportView } from "@/components/ReportView";
import { ScrapingVisualization } from "@/components/ScrapingVisualization";
import { AugmentationPanel } from "@/components/AugmentationPanel";
import { useToast } from "@/hooks/use-toast";
import { Sparkles } from "lucide-react";

interface ReportData {
  novelty_analysis: string;
  knowledge_graph: {
    nodes: any[];
    links: any[];
    stats: {
      total_nodes: number;
      total_links: number;
      target_paper: string;
      retrieved_papers: number;
      novelty_score: number;
    };
  };
  augmentation_metadata?: {
    additional_papers: number;
    gaps_identified: number;
    search_queries: number;
    total_papers: number;
  };
}

interface AugmentationState {
  isActive: boolean;
  progress: string;
  papersFound: number;
  gapsIdentified: number;
  currentPhase: string;
  newPapers: Array<{
    title: string;
    year?: number;
    authors?: string[];
  }>;
  initialNoveltyScore?: number;
  finalNoveltyScore?: number;
  currentNoveltyScore?: number; // Real-time updating score
}

const Index = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [reportData, setReportData] = useState<ReportData | null>(null);
  const [showVisualization, setShowVisualization] = useState(false);
  const [clientId, setClientId] = useState<string>("");
  const clientIdRef = useRef<string>("");
  const [augmentationState, setAugmentationState] = useState<AugmentationState | null>(null);
  const [isAugmenting, setIsAugmenting] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const { toast } = useToast();

  const handleFileUpload = async (file: File) => {
    setIsLoading(true);
    setReportData(null);
    setAugmentationState(null);
    setIsAugmenting(false);
    
    // Generate client ID for WebSocket connection
    const newClientId = `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    clientIdRef.current = newClientId;
    setClientId(newClientId);
    setShowVisualization(true);
    
    // Wait a moment for WebSocket to connect
    await new Promise(resolve => setTimeout(resolve, 500));

    // Setup WebSocket listener for augmentation updates
    setupWebSocketListener(newClientId);

    const formData = new FormData();
    formData.append("file", file);

    try {
      // Upload with client_id for WebSocket progress updates
      const response = await axios.post<ReportData>(
        `/api/v1/upload_paper/?client_id=${newClientId}`, 
        formData, 
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          timeout: 300000, // 5 minutes timeout for long-running analysis
        }
      );

      // Initial results are ready - show them immediately
      setReportData(response.data);
      setIsLoading(false);
      setShowVisualization(false);
      // Keep WebSocket open for augmentation updates
      
      // Initialize augmentation state to show waiting panel
      setAugmentationState({
        isActive: true,
        progress: "Waiting...",
        papersFound: 0,
        gapsIdentified: 0,
        currentPhase: "Initial analysis complete. Augmentation starting soon...",
        newPapers: [],
        initialNoveltyScore: response.data.knowledge_graph?.stats?.novelty_score
      });
      
      toast({
        title: "Initial Analysis Complete",
        description: "Knowledge graph is ready. Augmentation is running in the background.",
      });
    } catch (error) {
      console.error("Upload error:", error);
      setShowVisualization(false);
      setIsAugmenting(false);
      if (wsRef.current) {
        wsRef.current.close();
      }
      toast({
        title: "Analysis Failed",
        description: axios.isAxiosError(error) 
          ? error.response?.data?.message || "Failed to analyze research paper. Please try again."
          : "An unexpected error occurred.",
        variant: "destructive",
      });
      setIsLoading(false);
    }
  };

  const setupWebSocketListener = (clientId: string) => {
    // Close existing connection if any
    if (wsRef.current) {
      wsRef.current.close();
    }

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/api/v1/ws/scraping/${clientId}`;
    
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("WebSocket connected for augmentation updates");
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        handleAugmentationUpdate(message);
      } catch (error) {
        console.error("Error parsing WebSocket message:", error);
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    ws.onclose = () => {
      console.log("WebSocket disconnected");
    };
  };

  const handleAugmentationUpdate = (message: any) => {
    const { event, data } = message;

    switch (event) {
      case "initial_results_ready":
        // Initial results are ready - initialize augmentation state
        if (!augmentationState && reportData) {
          setAugmentationState({
            isActive: true,
            progress: "Waiting...",
            papersFound: 0,
            gapsIdentified: 0,
            currentPhase: "Initial analysis complete. Waiting for augmentation to start...",
            newPapers: [],
            initialNoveltyScore: reportData.knowledge_graph?.stats?.novelty_score
          });
        }
        break;

      case "augmentation_started":
        setIsAugmenting(true);
        setAugmentationState({
          isActive: true,
          progress: "Starting...",
          papersFound: 0,
          gapsIdentified: data.gaps_identified || 0,
          currentPhase: "Identifying gaps and interesting areas...",
          newPapers: [],
          initialNoveltyScore: reportData?.knowledge_graph?.stats?.novelty_score
        });
        break;

      case "augmentation_search_queries":
        setAugmentationState(prev => prev ? {
          ...prev,
          currentPhase: `Generating search queries...`,
          progress: "10%"
        } : null);
        break;

      case "augmentation_scraping":
        setAugmentationState(prev => prev ? {
          ...prev,
          currentPhase: data.message || `Scraping additional papers...`,
          progress: data.query_index ? `${30 + (data.query_index / (data.total_queries || 1)) * 20}%` : "30%"
        } : null);
        break;

      case "augmentation_paper_found":
        setAugmentationState(prev => prev ? {
          ...prev,
          papersFound: data.total_found || (prev.papersFound + 1),
          currentPhase: `Found: ${data.paper?.title?.substring(0, 40) || "paper"}...`,
          progress: "50%",
          newPapers: data.paper ? [...prev.newPapers, data.paper].slice(-10) : prev.newPapers
        } : null);
        break;

      case "augmentation_papers_found":
        setAugmentationState(prev => prev ? {
          ...prev,
          papersFound: data.additional_papers || prev.papersFound,
          currentPhase: `Found ${data.additional_papers || 0} additional papers`,
          progress: "60%"
        } : null);
        break;

      case "augmentation_parsing":
        setAugmentationState(prev => prev ? {
          ...prev,
          currentPhase: data.message || `Analyzing ${data.papers_count || (prev?.papersFound || 0)} papers...`,
          progress: "70%"
        } : null);
        break;

      case "augmentation_novelty":
        // Update current novelty score as it's being calculated
        if (data.novelty_score !== undefined) {
          setAugmentationState(prev => prev ? {
            ...prev,
            currentNoveltyScore: data.novelty_score,
            currentPhase: "Re-assessing novelty...",
            progress: "85%"
          } : null);
        } else {
          setAugmentationState(prev => prev ? {
            ...prev,
            currentPhase: "Re-assessing novelty...",
            progress: "85%"
          } : null);
        }
        break;

      case "knowledge_graph_updated":
        // Update the knowledge graph with new data
        if (data.knowledge_graph && reportData) {
          const updatedNoveltyScore = data.knowledge_graph?.stats?.novelty_score;
          setReportData({
            ...reportData,
            knowledge_graph: data.knowledge_graph
          });
          
          // Update current novelty score in augmentation state
          if (updatedNoveltyScore !== undefined) {
            setAugmentationState(prev => prev ? {
              ...prev,
              currentNoveltyScore: updatedNoveltyScore,
              currentPhase: "Knowledge graph updated",
              progress: "90%"
            } : null);
          }
          
          toast({
            title: "Graph Updated",
            description: `Knowledge graph updated with ${data.total_papers || 0} papers`,
          });
        } else {
          setAugmentationState(prev => prev ? {
            ...prev,
            currentPhase: "Knowledge graph updated",
            progress: "90%"
          } : null);
        }
        break;

      case "final_novelty_assessment":
        // Update novelty analysis with final results
        if (data.augmented_novelty_analysis && reportData) {
          setReportData({
            ...reportData,
            novelty_analysis: data.augmented_novelty_analysis,
            knowledge_graph: {
              ...reportData.knowledge_graph,
              stats: {
                ...reportData.knowledge_graph.stats,
                novelty_score: data.novelty_score || reportData.knowledge_graph.stats.novelty_score
              }
            }
          });
          toast({
            title: "Final Novelty Score",
            description: `Novelty score: ${data.novelty_score || "N/A"}/10`,
          });
        }
        setAugmentationState(prev => prev ? {
          ...prev,
          currentPhase: `Final novelty score: ${data.novelty_score || "N/A"}/10`,
          progress: "95%",
          finalNoveltyScore: data.novelty_score
        } : null);
        break;

      case "augmentation_complete":
        setAugmentationState(prev => prev ? {
          ...prev,
          currentPhase: "Augmentation complete",
          progress: "Complete",
          papersFound: data.additional_papers || prev.papersFound,
          finalNoveltyScore: data.final_novelty_score
        } : null);
        toast({
          title: "Augmentation Complete",
          description: `Found ${data.additional_papers || 0} additional papers. Final novelty score: ${data.final_novelty_score || "N/A"}/10`,
        });
        // Keep panel visible but mark as complete
        setTimeout(() => {
          setIsAugmenting(false);
        }, 10000);
        break;

      case "processing_complete":
        setIsAugmenting(false);
        if (wsRef.current) {
          wsRef.current.close();
        }
        break;

      case "augmentation_error":
        setAugmentationState(prev => prev ? {
          ...prev,
          currentPhase: `Error: ${data.error || "Unknown error"}`,
          progress: "Error"
        } : null);
        setIsAugmenting(false);
        toast({
          title: "Augmentation Error",
          description: data.error || "Augmentation failed, using initial results",
          variant: "destructive",
        });
        break;
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
            Analyze research papers for novelty, industry applications, and comprehensive insights
          </p>
        </div>

        {/* Main Content */}
        <div className="space-y-8">
          {!isLoading && !reportData && !showVisualization && (
            <FileUpload onFileSelect={handleFileUpload} isLoading={isLoading} />
          )}

          {showVisualization && clientId && (
            <ScrapingVisualization 
              clientId={clientId}
              onComplete={() => {
                setShowVisualization(false);
              }}
            />
          )}

          {isLoading && !showVisualization && <LoadingState />}

          {reportData && !isLoading && (
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
              {/* Main Report View - Takes 2/3 of the space */}
              <div className="xl:col-span-2 order-1">
                <ReportView 
                  novelty_analysis={reportData.novelty_analysis} 
                  knowledge_graph={reportData.knowledge_graph} 
                />
                <div className="flex justify-center mt-4">
                  <button
                    onClick={() => {
                      setReportData(null);
                      setShowVisualization(false);
                      setClientId("");
                      setAugmentationState(null);
                      setIsAugmenting(false);
                      if (wsRef.current) {
                        wsRef.current.close();
                      }
                    }}
                    className="text-sm text-muted-foreground hover:text-primary transition-colors underline"
                  >
                    Analyze another paper
                  </button>
                </div>
              </div>

              {/* Augmentation Panel - Takes 1/3 of the space on the side */}
              <div className="xl:col-span-1 order-2">
                {augmentationState ? (
                  <AugmentationPanel
                    isActive={augmentationState.isActive}
                    progress={augmentationState.progress}
                    papersFound={augmentationState.papersFound}
                    gapsIdentified={augmentationState.gapsIdentified}
                    currentPhase={augmentationState.currentPhase}
                    newPapers={augmentationState.newPapers}
                    initialNoveltyScore={augmentationState.initialNoveltyScore ?? reportData?.knowledge_graph?.stats?.novelty_score}
                    finalNoveltyScore={augmentationState.finalNoveltyScore}
                    currentNoveltyScore={augmentationState.currentNoveltyScore}
                  />
                ) : (
                  <AugmentationPanel
                    isActive={true}
                    progress="Waiting..."
                    papersFound={0}
                    gapsIdentified={0}
                    currentPhase="Initial analysis complete. Augmentation will start shortly..."
                    newPapers={[]}
                    initialNoveltyScore={reportData?.knowledge_graph?.stats?.novelty_score}
                  />
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Index;
