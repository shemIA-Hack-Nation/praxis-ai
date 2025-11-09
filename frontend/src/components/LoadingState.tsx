import { Loader2, Brain, Sparkles } from "lucide-react";
import { Card } from "@/components/ui/card";

export const LoadingState = () => {
  return (
    <Card className="p-12 bg-gradient-subtle border-border">
      <div className="flex flex-col items-center gap-6">
        <div className="relative">
          <Brain className="w-20 h-20 text-primary animate-pulse" />
          <Sparkles className="w-8 h-8 text-primary absolute -top-2 -right-2 animate-bounce" />
        </div>
        
        <div className="text-center space-y-2">
          <h3 className="text-2xl font-bold text-foreground">AI Agents at Work</h3>
          <p className="text-muted-foreground">
            Analyzing your notebook and generating comprehensive insights...
          </p>
        </div>

        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="w-4 h-4 animate-spin text-primary" />
          <span>This may take a few moments</span>
        </div>

        <div className="w-full max-w-md mt-4">
          <div className="h-2 bg-secondary rounded-full overflow-hidden">
            <div className="h-full bg-gradient-primary animate-pulse" style={{ width: "60%" }} />
          </div>
        </div>
      </div>
    </Card>
  );
};
