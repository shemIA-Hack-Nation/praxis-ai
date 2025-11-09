import React from 'react';

interface TerminalOutputProps {
  text: string;
  className?: string;
}

const TerminalOutput: React.FC<TerminalOutputProps> = ({ text, className = '' }) => {
  const formatTerminalText = (input: string): string => {
    return input
      // Remove ANSI color codes
      .replace(/\x1b\[[0-9;]*m/g, '')
      // Replace progress bar characters with simple characters
      .replace(/━/g, '█')
      // Clean up any remaining escape sequences
      .replace(/\[1m|\[0m|\[32m|\[37m/g, '');
  };

  const formatTrainingOutput = (text: string): React.ReactNode[] => {
    const lines = text.split('\n').filter(line => line.trim());
    
    return lines.map((line, index) => {
      // Check if it's an epoch line
      if (line.includes('Epoch ')) {
        return (
          <div key={index} className="text-cyan-400 font-semibold mb-1">
            {line}
          </div>
        );
      }
      
      // Check if it's a progress line with metrics
      if (line.includes('/') && (line.includes('accuracy') || line.includes('loss'))) {
        const parts = line.split(' - ');
        const progressPart = parts[0];
        const metricsPart = parts.slice(1).join(' - ');
        
        return (
          <div key={index} className="mb-2">
            <div className="text-white mb-1">{progressPart}</div>
            {metricsPart && (
              <div className="text-yellow-300 text-sm ml-4">
                {metricsPart.split(' - ').map((metric, i) => (
                  <span key={i} className="mr-4">
                    {metric}
                  </span>
                ))}
              </div>
            )}
          </div>
        );
      }
      
      // Default formatting for other lines
      return (
        <div key={index} className="text-gray-300 mb-1">
          {line}
        </div>
      );
    });
  };

  const cleanedText = formatTerminalText(text);

  return (
    <div className={`font-mono text-sm bg-black/80 p-4 rounded border border-gray-700 ${className}`}>
      <div className="space-y-1">
        {formatTrainingOutput(cleanedText)}
      </div>
    </div>
  );
};

export default TerminalOutput;