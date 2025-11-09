/**
 * Real-time visualization component for paper scraping progress.
 * Shows papers as they are found during the scraping process.
 */
import { useEffect, useState, useRef } from "react";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Search, FileText, CheckCircle2, Loader2 } from "lucide-react";

interface Paper {
  title: string;
  year?: number;
  authors?: string[];
  categories?: string[];
}

interface ScrapingProgress {
  event: string;
  data: any;
  timestamp: string;
}

interface ScrapingVisualizationProps {
  clientId: string;
  onComplete?: () => void;
}

export const ScrapingVisualization = ({ clientId, onComplete }: ScrapingVisualizationProps) => {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [currentPhase, setCurrentPhase] = useState<string>("");
  const [progress, setProgress] = useState(0);
  const [keywords, setKeywords] = useState<string[]>([]);
  const [methods, setMethods] = useState<string[]>([]);
  const [categories, setCategories] = useState<string[]>([]);
  const [totalPapers, setTotalPapers] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Connect to WebSocket
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/api/v1/ws/scraping/${clientId}`;
    
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      console.log("WebSocket connected for scraping visualization");
    };

    ws.onmessage = (event) => {
      try {
        const message: ScrapingProgress = JSON.parse(event.data);
        handleProgressUpdate(message);
      } catch (error) {
        console.error("Error parsing WebSocket message:", error);
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    ws.onclose = () => {
      setIsConnected(false);
      console.log("WebSocket disconnected");
    };

    // Cleanup on unmount
    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [clientId]);

  const handleProgressUpdate = (message: ScrapingProgress) => {
    const { event, data } = message;

    switch (event) {
      case "extraction_started":
        setCurrentPhase("Extracting content from PDF...");
        setProgress(5);
        break;

      case "pdf_parsed":
        setCurrentPhase("PDF parsed, extracting structured content...");
        setProgress(10);
        break;

      case "content_extracted":
        setCurrentPhase(`Content extracted: ${data.title || "Paper"}`);
        setProgress(15);
        if (data.publication_date) {
          setCurrentPhase(`Paper from ${data.publication_date} - ${data.title || "Paper"}`);
        }
        break;

      case "search_term_extraction_started":
        setCurrentPhase("Extracting search terms with LLM...");
        setProgress(20);
        break;

      case "search_term_extraction_complete":
        setCurrentPhase("Search terms extracted");
        setKeywords(data.keywords || []);
        setMethods(data.methods || []);
        setCategories(data.categories || []);
        setProgress(25);
        break;

      case "search_started":
        setCurrentPhase(`Searching arXiv: ${data.query_terms?.join(", ") || ""}`);
        setProgress(30);
        break;

      case "paper_found":
        setCurrentPhase(`Found paper ${data.count}/${data.total || "?"}`);
        if (data.paper) {
          setPapers(prev => {
            // Avoid duplicates
            const exists = prev.some(p => p.title === data.paper.title);
            if (!exists) {
              return [...prev, data.paper];
            }
            return prev;
          });
          setTotalPapers(data.total || papers.length + 1);
          // Update progress based on papers found (30-50%)
          if (data.total) {
            setProgress(30 + (data.count / data.total) * 20);
          }
        }
        break;

      case "search_complete":
        setCurrentPhase(`Search complete: ${data.papers_found || 0} papers found`);
        setProgress(50);
        break;

      case "ranking_complete":
        setCurrentPhase("Ranking and filtering papers...");
        setProgress(55);
        break;

      case "indexing_started":
        setCurrentPhase(`Generating embeddings for ${data.papers_count || 0} papers...`);
        setProgress(60);
        break;

      case "indexing_complete":
        setCurrentPhase("Indexes built successfully");
        setProgress(75);
        break;

      case "novelty_assessment_started":
        setCurrentPhase("Assessing novelty based on similarity and date context...");
        setProgress(80);
        break;

      case "novelty_assessment_complete":
        setCurrentPhase(`Novelty score: ${data.novelty_score || "N/A"}/10`);
        setProgress(85);
        break;

      case "report_generation_started":
        setCurrentPhase("Generating final reports...");
        setProgress(90);
        break;

      case "report_generation_complete":
        setCurrentPhase("Initial analysis complete!");
        setProgress(85);
        break;

      case "augmentation_started":
        setCurrentPhase("Augmentation: Identifying gaps and interesting areas...");
        setProgress(86);
        break;

      case "augmentation_search_queries":
        setCurrentPhase(`Augmentation: Generating search queries from ${data.message || ""}...`);
        setProgress(87);
        break;

      case "augmentation_scraping":
        setCurrentPhase(`Augmentation: Scraping additional papers with ${data.message || ""}...`);
        setProgress(88);
        break;

      case "augmentation_parsing":
        setCurrentPhase(`Augmentation: Re-parsing ${data.message || ""} papers...`);
        setProgress(92);
        break;

      case "augmentation_novelty":
        setCurrentPhase("Augmentation: Re-assessing novelty with augmented corpus...");
        setProgress(95);
        break;

      case "augmentation_report_generation":
        setCurrentPhase("Generating final reports with augmented data...");
        setProgress(97);
        break;

      case "augmentation_complete":
        setCurrentPhase(`Augmentation complete: ${data.additional_papers || 0} additional papers processed`);
        setProgress(99);
        break;

      case "processing_complete":
        setCurrentPhase("All processing complete!");
        setProgress(100);
        if (onComplete) {
          setTimeout(onComplete, 1000);
        }
        break;

      case "error":
        setCurrentPhase(`Error: ${data.error || "Unknown error"}`);
        break;

      default:
        console.log("Unknown event:", event, data);
    }
  };

  return (
    <Card className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isConnected ? (
            <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />
          ) : (
            <div className="w-3 h-3 bg-gray-400 rounded-full" />
          )}
          <h3 className="text-lg font-semibold">Real-time Paper Scraping</h3>
        </div>
        <Badge variant={isConnected ? "default" : "secondary"}>
          {isConnected ? "Connected" : "Disconnected"}
        </Badge>
      </div>

      {/* Progress Bar */}
      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span className="text-muted-foreground">{currentPhase}</span>
          <span className="text-muted-foreground">{Math.round(progress)}%</span>
        </div>
        <Progress value={progress} className="h-2" />
      </div>

      {/* Search Terms */}
      {(keywords.length > 0 || methods.length > 0 || categories.length > 0) && (
        <div className="space-y-3">
          <h4 className="text-sm font-semibold">Extracted Search Terms</h4>
          {keywords.length > 0 && (
            <div>
              <p className="text-xs text-muted-foreground mb-1">Keywords:</p>
              <div className="flex flex-wrap gap-1">
                {keywords.slice(0, 10).map((kw, i) => (
                  <Badge key={i} variant="outline" className="text-xs">
                    {kw}
                  </Badge>
                ))}
              </div>
            </div>
          )}
          {methods.length > 0 && (
            <div>
              <p className="text-xs text-muted-foreground mb-1">Methods:</p>
              <div className="flex flex-wrap gap-1">
                {methods.slice(0, 5).map((method, i) => (
                  <Badge key={i} variant="secondary" className="text-xs">
                    {method}
                  </Badge>
                ))}
              </div>
            </div>
          )}
          {categories.length > 0 && (
            <div>
              <p className="text-xs text-muted-foreground mb-1">Categories:</p>
              <div className="flex flex-wrap gap-1">
                {categories.map((cat, i) => (
                  <Badge key={i} variant="default" className="text-xs">
                    {cat}
                  </Badge>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Papers Found */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-semibold">Papers Found</h4>
          <Badge variant="outline">
            {papers.length}{totalPapers > 0 ? ` / ${totalPapers}` : ""}
          </Badge>
        </div>
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {papers.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <Search className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">Waiting for papers...</p>
            </div>
          ) : (
            papers.map((paper, index) => (
              <Card key={index} className="p-3 border-l-4 border-l-primary">
                <div className="flex items-start gap-2">
                  <FileText className="w-4 h-4 text-primary mt-0.5 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">{paper.title}</p>
                    <div className="flex items-center gap-2 mt-1">
                      {paper.year && (
                        <Badge variant="outline" className="text-xs">
                          {paper.year}
                        </Badge>
                      )}
                      {paper.categories && paper.categories.length > 0 && (
                        <Badge variant="secondary" className="text-xs">
                          {paper.categories[0]}
                        </Badge>
                      )}
                    </div>
                    {paper.authors && paper.authors.length > 0 && (
                      <p className="text-xs text-muted-foreground mt-1">
                        {paper.authors.slice(0, 2).join(", ")}
                        {paper.authors.length > 2 && " et al."}
                      </p>
                    )}
                  </div>
                  <CheckCircle2 className="w-4 h-4 text-green-500 flex-shrink-0" />
                </div>
              </Card>
            ))
          )}
        </div>
      </div>
    </Card>
  );
};

