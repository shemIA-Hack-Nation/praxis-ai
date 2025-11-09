import { useState, useCallback } from "react";
import { Upload, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  isLoading?: boolean;
}

export const FileUpload = ({ onFileSelect, isLoading }: FileUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragging(true);
    } else if (e.type === "dragleave") {
      setIsDragging(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    const pdfFile = files.find(file => file.name.endsWith('.pdf'));
    
    if (pdfFile) {
      setSelectedFile(pdfFile);
    }
  }, []);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      setSelectedFile(files[0]);
    }
  }, []);

  const handleGenerate = () => {
    if (selectedFile) {
      onFileSelect(selectedFile);
    }
  };

  return (
    <Card className="p-8 bg-card border-border transition-smooth">
      <div
        className={`border-2 border-dashed rounded-lg p-12 text-center transition-all ${
          isDragging 
            ? "border-primary bg-primary/10 scale-[1.02]" 
            : "border-border hover:border-primary/50"
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <div className="flex flex-col items-center gap-4">
          {selectedFile ? (
            <>
              <FileText className="w-16 h-16 text-primary" />
              <div>
                <p className="text-lg font-semibold text-foreground">{selectedFile.name}</p>
                <p className="text-sm text-muted-foreground mt-1">
                  {(selectedFile.size / 1024).toFixed(2)} KB
                </p>
              </div>
              <div className="flex gap-3 mt-4">
                <Button
                  onClick={handleGenerate}
                  disabled={isLoading}
                  className="bg-gradient-primary text-primary-foreground hover:opacity-90 shadow-glow"
                >
                  {isLoading ? "Analyzing..." : "Analyze Paper"}
                </Button>
                <Button
                  onClick={() => setSelectedFile(null)}
                  variant="secondary"
                  disabled={isLoading}
                >
                  Change File
                </Button>
              </div>
            </>
          ) : (
            <>
              <Upload className="w-16 h-16 text-muted-foreground" />
              <div>
                <p className="text-lg font-semibold text-foreground mb-2">
                  Drop your Research Paper here
                </p>
                <p className="text-sm text-muted-foreground">
                  or click to browse for .pdf files
                </p>
              </div>
              <label className="mt-4">
                <input
                  type="file"
                  accept=".pdf"
                  onChange={handleFileInput}
                  className="hidden"
                  disabled={isLoading}
                />
                <Button 
                  variant="secondary" 
                  className="cursor-pointer"
                  disabled={isLoading}
                  asChild
                >
                  <span>Select File</span>
                </Button>
              </label>
            </>
          )}
        </div>
      </div>
    </Card>
  );
};
