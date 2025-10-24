import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, File, X, CheckCircle2, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { motion, AnimatePresence } from "framer-motion";
import { apiClient, CIFProcessingResult } from "@/lib/api-client";

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  onStructureProcessed?: (result: CIFProcessingResult) => void;
  acceptedFileTypes?: Record<string, string[]>;
  maxSize?: number;
}

const FileUpload = ({ 
  onFileSelect, 
  onStructureProcessed,
  acceptedFileTypes = { 'chemical/x-cif': ['.cif'] },
  maxSize = 5 * 1024 * 1024 // 5MB default
}: FileUploadProps) => {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const [processingResult, setProcessingResult] = useState<CIFProcessingResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    setIsValidating(true);
    setError(null);
    setProcessingResult(null);

    // Basic client-side validation
    if (!file.name.endsWith('.cif')) {
      toast.error("Invalid file type. Please upload a CIF file.");
      setIsValidating(false);
      setError("Invalid file type");
      return;
    }

    try {
      // Process CIF file with backend
      const result = await apiClient.processCIF(file);
      
      if (result.success) {
        setUploadedFile(file);
        setProcessingResult(result);
        onFileSelect(file);
        onStructureProcessed?.(result);
        toast.success(`CIF file "${file.name}" processed successfully`);
      } else {
        // Handle case where CIF is parsed but can't be processed for model
        setUploadedFile(file);
        setError(result.error || "Failed to process CIF file for model inference");
        toast.error(result.error || "Failed to process CIF file for model inference");
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to process CIF file";
      setError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setIsValidating(false);
    }
  }, [onFileSelect, onStructureProcessed]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: acceptedFileTypes,
    maxSize,
    multiple: false,
  });

  const removeFile = () => {
    setUploadedFile(null);
    setProcessingResult(null);
    setError(null);
  };

  return (
    <div className="space-y-4">
      <div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-xl p-12 text-center cursor-pointer
          transition-all duration-300 ease-in-out
          ${isDragActive 
            ? 'border-primary bg-primary/5 scale-[1.02]' 
            : 'border-border hover:border-primary/50 hover:bg-muted/30'
          }
          ${uploadedFile ? 'bg-muted/20' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        <AnimatePresence mode="wait">
          {!uploadedFile ? (
            <motion.div
              key="upload-prompt"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="flex flex-col items-center gap-4"
            >
              <div className="p-4 rounded-full bg-primary/10">
                <Upload className="h-10 w-10 text-primary" />
              </div>
              <div>
                <p className="text-lg font-medium mb-1">
                  {isDragActive ? "Drop your CIF file here" : "Upload CIF File"}
                </p>
                <p className="text-sm text-muted-foreground">
                  Drag and drop or click to browse (max 5MB)
                </p>
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="file-info"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="flex items-center justify-between gap-4"
            >
              <div className="flex items-center gap-3">
                <div className="p-3 rounded-lg bg-primary/10">
                  <File className="h-6 w-6 text-primary" />
                </div>
                <div className="text-left">
                  <p className="font-medium">{uploadedFile.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {(uploadedFile.size / 1024).toFixed(2)} KB
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-green-600" />
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    removeFile();
                  }}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {isValidating && (
          <div className="absolute inset-0 bg-background/80 backdrop-blur-sm rounded-xl flex items-center justify-center">
            <div className="flex flex-col items-center gap-2">
              <div className="h-8 w-8 border-4 border-primary border-t-transparent rounded-full animate-spin" />
              <p className="text-sm font-medium">Validating file...</p>
            </div>
          </div>
        )}
      </div>

      {/* Structure Information Display */}
      {processingResult && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-green-50 border border-green-200 rounded-lg p-4"
        >
          <div className="flex items-center gap-2 mb-3">
            <CheckCircle2 className="h-5 w-5 text-green-600" />
            <h3 className="font-semibold text-green-800">Structure Information</h3>
          </div>
          
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="font-medium text-gray-700">Formula:</span>
              <span className="ml-2 text-gray-900">{processingResult.structure_info.formula}</span>
            </div>
            <div>
              <span className="font-medium text-gray-700">Atoms:</span>
              <span className="ml-2 text-gray-900">{processingResult.structure_info.num_atoms}</span>
            </div>
            <div>
              <span className="font-medium text-gray-700">Elements:</span>
              <span className="ml-2 text-gray-900">{processingResult.structure_info.num_elements}</span>
            </div>
            <div>
              <span className="font-medium text-gray-700">Space Group:</span>
              <span className="ml-2 text-gray-900">{processingResult.structure_info.space_group_symbol} ({processingResult.structure_info.space_group_number})</span>
            </div>
            <div>
              <span className="font-medium text-gray-700">Density:</span>
              <span className="ml-2 text-gray-900">{processingResult.structure_info.density.toFixed(2)} g/cm³</span>
            </div>
            <div>
              <span className="font-medium text-gray-700">Volume:</span>
              <span className="ml-2 text-gray-900">{processingResult.structure_info.volume.toFixed(2)} Å³</span>
            </div>
          </div>
          
          <div className="mt-3 pt-3 border-t border-green-200">
            <div className="flex items-center gap-2 text-sm">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span className="text-green-700">Ready for prediction ({processingResult.num_atoms} atoms, {processingResult.num_edges} bonds)</span>
            </div>
          </div>
        </motion.div>
      )}

      {/* Error Display */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-red-50 border border-red-200 rounded-lg p-4"
        >
          <div className="flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-red-600" />
            <h3 className="font-semibold text-red-800">Processing Error</h3>
          </div>
          <p className="text-red-700 text-sm mt-1">{error}</p>
        </motion.div>
      )}
    </div>
  );
};

export default FileUpload;
