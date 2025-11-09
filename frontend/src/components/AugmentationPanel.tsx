/**
 * Augmentation Panel Component
 * Shows real-time progress of the augmentation process
 */
import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Sparkles, Search, FileText, CheckCircle2, Loader2, Activity, TrendingUp } from "lucide-react";

interface AugmentationPanelProps {
  isActive: boolean;
  progress: string;
  papersFound: number;
  gapsIdentified: number;
  currentPhase: string;
  newPapers?: Array<{
    title: string;
    year?: number;
    authors?: string[];
  }>;
  initialNoveltyScore?: number;
  finalNoveltyScore?: number;
  currentNoveltyScore?: number; // Real-time updating score during augmentation
}

export const AugmentationPanel = ({
  isActive,
  progress,
  papersFound,
  gapsIdentified,
  currentPhase,
  newPapers = [],
  initialNoveltyScore,
  finalNoveltyScore,
  currentNoveltyScore
}: AugmentationPanelProps) => {
  const [scrapingEffect, setScrapingEffect] = useState(false);
  const [paperCount, setPaperCount] = useState(0);

  // Scraping animation effect
  useEffect(() => {
    if (progress !== "Waiting..." && progress !== "Complete" && progress !== "Error" && progress !== "Starting...") {
      setScrapingEffect(true);
      const interval = setInterval(() => {
        setPaperCount(prev => prev + 1);
      }, 800);
      return () => {
        clearInterval(interval);
        setScrapingEffect(false);
      };
    }
  }, [progress, papersFound]);

  // Update paper count when papers are actually found
  useEffect(() => {
    if (papersFound > 0) {
      setPaperCount(papersFound);
    }
  }, [papersFound]);

  const progressValue = progress === "Complete" ? 100 : 
                       progress === "Error" ? 0 :
                       progress === "Waiting..." ? 0 :
                       progress === "Starting..." ? 5 :
                       parseInt(progress) || 0;
  
  const isWaiting = progress === "Waiting...";
  const isComplete = progress === "Complete";
  const isError = progress === "Error";
  const isScraping = !isWaiting && !isComplete && !isError && progressValue > 0;
  
  // Determine current displayed novelty score
  const displayedNoveltyScore = finalNoveltyScore !== undefined 
    ? finalNoveltyScore 
    : currentNoveltyScore !== undefined 
    ? currentNoveltyScore 
    : initialNoveltyScore;

  return (
    <Card className="p-4 bg-secondary/50 border-2 border-primary/20 shadow-lg sticky top-4 min-h-[400px]">
      <div className="flex items-center gap-2 mb-4">
        <Sparkles className={`w-5 h-5 text-primary ${!isComplete && !isWaiting ? 'animate-pulse' : ''}`} />
        <h3 className="text-lg font-semibold">Augmentation Agent</h3>
        {papersFound > 0 && (
          <Badge variant="outline" className="ml-auto bg-primary/10">
            {papersFound} papers
          </Badge>
        )}
      </div>

      <div className="space-y-4">
        {/* Scraping Activity Effect */}
        {isScraping && (
          <Card className="p-4 bg-primary/5 border-2 border-primary/30 animate-pulse">
            <div className="flex items-center gap-3">
              <div className="relative">
                <Activity className="w-5 h-5 text-primary animate-pulse" />
                <div className="absolute inset-0 w-5 h-5 border-2 border-primary rounded-full animate-ping opacity-75"></div>
              </div>
              <div className="flex-1">
                <div className="text-sm font-semibold text-primary mb-1">Actively Scraping Papers</div>
                <div className="text-xs text-muted-foreground">
                  {scrapingEffect && (
                    <span className="inline-block animate-bounce">
                      Searching arXiv • Analyzing papers • Extracting concepts
                    </span>
                  )}
                </div>
              </div>
            </div>
            <div className="mt-3 flex items-center gap-2">
              <div className="flex-1 h-1.5 bg-primary/20 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-primary rounded-full transition-all duration-500 animate-pulse"
                  style={{ width: `${Math.min(progressValue, 100)}%` }}
                />
              </div>
              <span className="text-xs text-muted-foreground font-medium">{progress}</span>
            </div>
          </Card>
        )}

        <div>
          <div className="flex justify-between text-sm mb-2">
            <span className="text-muted-foreground truncate mr-2">{currentPhase}</span>
            <span className="text-muted-foreground whitespace-nowrap">{progress}</span>
          </div>
          <Progress value={progressValue} className="h-2" />
        </div>

        {gapsIdentified > 0 && (
          <div className="text-sm text-muted-foreground">
            <span className="font-medium">Gaps identified:</span> {gapsIdentified}
          </div>
        )}

        {/* Novelty Score Section - Always show if we have any score data */}
        {(initialNoveltyScore !== undefined || currentNoveltyScore !== undefined || finalNoveltyScore !== undefined) && (
          <Card className="p-4 bg-gradient-to-br from-primary/10 to-primary/5 border-2 border-primary/20">
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-primary" />
                <div className="text-sm font-semibold">Novelty Score</div>
              </div>
              
              {/* Initial Score */}
              {initialNoveltyScore !== undefined && (
                <div className="flex items-center justify-between p-2 bg-background/50 rounded">
                  <span className="text-xs text-muted-foreground">Initial:</span>
                  <span className="text-sm font-semibold">{initialNoveltyScore.toFixed(1)}/10</span>
                </div>
              )}

              {/* Current/Updating Score - Show during scraping */}
              {isScraping && (currentNoveltyScore !== undefined || finalNoveltyScore !== undefined) && (
                <div className="flex items-center justify-between p-3 bg-primary/20 rounded border-2 border-primary/30 animate-pulse">
                  <div className="flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin text-primary" />
                    <span className="text-xs text-muted-foreground">Updating:</span>
                  </div>
                  <span className="text-lg font-bold text-primary animate-pulse">
                    {(currentNoveltyScore ?? finalNoveltyScore ?? initialNoveltyScore ?? 0).toFixed(1)}/10
                  </span>
                </div>
              )}

              {/* Final Score */}
              {isComplete && finalNoveltyScore !== undefined && (
                <div className="flex items-center justify-between p-3 bg-primary/30 rounded border-2 border-primary">
                  <span className="text-xs font-semibold">Final:</span>
                  <span className="text-xl font-bold text-primary">
                    {finalNoveltyScore.toFixed(1)}/10
                  </span>
                </div>
              )}

              {/* Score Change Indicator */}
              {finalNoveltyScore !== undefined && initialNoveltyScore !== undefined && (
                <div className="text-xs text-center pt-2 border-t border-border/50">
                  {finalNoveltyScore > initialNoveltyScore ? (
                    <span className="text-green-500 font-semibold">
                      ↑ Increased by {(finalNoveltyScore - initialNoveltyScore).toFixed(1)} points
                    </span>
                  ) : finalNoveltyScore < initialNoveltyScore ? (
                    <span className="text-orange-500 font-semibold">
                      ↓ Decreased by {(initialNoveltyScore - finalNoveltyScore).toFixed(1)} points
                    </span>
                  ) : (
                    <span className="text-muted-foreground">No change</span>
                  )}
                </div>
              )}

              {!isComplete && !isScraping && initialNoveltyScore !== undefined && (
                <div className="text-xs text-center text-muted-foreground italic pt-2">
                  Score will update as papers are processed
                </div>
              )}
            </div>
          </Card>
        )}

        {newPapers.length > 0 && (
          <div className="space-y-2">
            <div className="text-sm font-medium">Recent Papers Found:</div>
            <div className="space-y-1 max-h-48 overflow-y-auto">
              {newPapers.slice(-5).map((paper, idx) => (
                <div key={idx} className="text-xs p-2 bg-background/50 rounded border border-border/50">
                  <div className="font-medium truncate">{paper.title}</div>
                  {paper.year && (
                    <div className="text-muted-foreground">Year: {paper.year}</div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {isWaiting && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>Waiting for augmentation to start...</span>
          </div>
        )}

        {isComplete && (
          <div className="flex items-center gap-2 text-sm text-primary">
            <CheckCircle2 className="w-4 h-4" />
            <span>Augmentation complete</span>
          </div>
        )}

        {isError && (
          <div className="flex items-center gap-2 text-sm text-destructive">
            <span>⚠️</span>
            <span>Augmentation error occurred</span>
          </div>
        )}
      </div>
    </Card>
  );
};

