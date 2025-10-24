import { useState } from "react";
import { motion } from "framer-motion";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Navigation from "@/components/Navigation";
import FileUpload from "@/components/FileUpload";
import StructureViewer from "@/components/StructureViewer";
import { toast } from "sonner";
import { useNavigate } from "react-router-dom";
import { apiClient, CIFProcessingResult } from "@/lib/api-client";

const Predict = () => {
  const navigate = useNavigate();
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [structureData, setStructureData] = useState<any>(null);
  const [cifProcessingResult, setCifProcessingResult] = useState<CIFProcessingResult | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleFileSelect = (file: File) => {
    setUploadedFile(file);
  };

  const handleStructureProcessed = (result: CIFProcessingResult) => {
    setCifProcessingResult(result);
  };

  const handleStructureSubmit = (data: any) => {
    setStructureData(data);
  };

  const handleSubmitJob = async () => {
    if (!uploadedFile && !structureData) {
      toast.error("Please upload a CIF file or input structure data");
      return;
    }
    setIsSubmitting(true);

    try {
      const payload: any = {};
      if (uploadedFile) {
        payload.file = uploadedFile;
      }
      if (structureData) {
        payload.structureData = structureData;
      }
      const response = await apiClient.submitJob(payload);
      // Navigate to results page with jobId
      navigate(`/results/${response.jobId}`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Failed to submit job";
      toast.error(errorMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <div className="container mx-auto px-4 py-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-5xl mx-auto"
        >
          <div className="mb-8">
            <h1 className="text-4xl font-bold mb-3">Submit Prediction Job</h1>
            <p className="text-muted-foreground text-lg">
              Upload a CIF file for prediction
            </p>
          </div>

          <div className="grid lg:grid-cols-3 gap-6">
            {/* Left column - Input */}
            <div className="lg:col-span-2 space-y-6">
              <Tabs defaultValue="upload" className="w-full">
                <TabsList className="w-full">
                  <TabsTrigger value="upload">CIF Upload</TabsTrigger>
                </TabsList>
                <TabsContent value="upload" className="space-y-6 mt-6">
                  <FileUpload 
                    onFileSelect={handleFileSelect} 
                    onStructureProcessed={handleStructureProcessed}
                  />
                </TabsContent>
              </Tabs>

              <Button 
                onClick={handleSubmitJob} 
                size="lg" 
                className="w-full gradient-primary"
                disabled={isSubmitting || (!uploadedFile && !structureData)}
              >
                {isSubmitting ? "Submitting..." : "Submit for Prediction"}
              </Button>
            </div>

            {/* Right column - 3D Preview */}
            <div className="lg:col-span-1">
              <Card className="p-6 sticky top-24">
                <h3 className="text-lg font-semibold mb-4">Structure Preview</h3>
                <StructureViewer 
                  hasStructure={!!(uploadedFile || structureData)} 
                  structureInfo={cifProcessingResult?.structure_info}
                />
              </Card>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Predict;
