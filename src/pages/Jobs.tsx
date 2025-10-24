import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import Navigation from "@/components/Navigation";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Eye, Download, Clock, CheckCircle2, AlertCircle } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { apiClient } from "@/lib/api-client";

interface Job {
  id: string;
  sampleId: string;
  status: "pending" | "processing" | "completed" | "failed";
  submittedAt: string;
  completedAt?: string;
  jobName?: string;
}

const Jobs = () => {
  const navigate = useNavigate();
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedJob, setSelectedJob] = useState<any>(null);
  const [showModal, setShowModal] = useState(false);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    const fetchJobs = async () => {
      setLoading(true);
      try {
        const response = await apiClient.getJobs(); // Assumes apiClient.getJobs() returns job list
        setJobs(response || []);
      } catch (error) {
        // Optionally handle error
      } finally {
        setLoading(false);
      }
    };
    fetchJobs();
    interval = setInterval(fetchJobs, 5000); // Poll every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = (status: Job["status"]) => {
    switch (status) {
      case "completed":
        return <CheckCircle2 className="h-4 w-4" />;
      case "processing":
        return <Clock className="h-4 w-4 animate-spin" />;
      case "pending":
        return <Clock className="h-4 w-4" />;
      case "failed":
        return <AlertCircle className="h-4 w-4" />;
    }
  };

  const getStatusColor = (status: Job["status"]) => {
    switch (status) {
      case "completed":
        return "bg-green-500/10 text-green-700 dark:text-green-400";
      case "processing":
        return "bg-blue-500/10 text-blue-700 dark:text-blue-400";
      case "pending":
        return "bg-yellow-500/10 text-yellow-700 dark:text-yellow-400";
      case "failed":
        return "bg-red-500/10 text-red-700 dark:text-red-400";
    }
  };

  const handleViewResult = (job: any) => {
    setSelectedJob(job);
    setShowModal(true);
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
            <h1 className="text-4xl font-bold mb-3">My Prediction Jobs</h1>
            <p className="text-muted-foreground text-lg">
              Track and manage your crystal property prediction jobs
            </p>
          </div>

          {/* Render jobs list with status */}
          <div>
            {loading ? (
              <div>Loading jobs...</div>
            ) : jobs.length === 0 ? (
              <div>No jobs found.</div>
            ) : (
              <div className="space-y-4">
                {jobs.map((job, index) => (
                  <motion.div
                    key={job.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                  >
                    <Card className="p-6 hover:shadow-md transition-shadow">
                      <div className="flex items-center justify-between gap-4 flex-wrap">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-3 mb-2">
                            <h3 className="font-semibold text-lg truncate">{job.sampleId}</h3>
                            <Badge className={`${getStatusColor(job.status)} flex items-center gap-1`}>
                              {getStatusIcon(job.status)}
                              {job.status}
                            </Badge>
                          </div>
                          <div className="text-sm text-muted-foreground space-y-1">
                            <p>Job ID: {job.id}</p>
                            <p>Submitted: {job.submittedAt}</p>
                            {job.completedAt && <p>Completed: {job.completedAt}</p>}
                            <p>
                              <strong>{job.jobName || job.id}</strong>
                            </p>
                          </div>
                        </div>

                        <div className="flex gap-2">
                          {job.status === "completed" && (
                            <>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => navigate(`/results/${job.id}`)}
                              >
                                <Eye className="h-4 w-4 mr-1" />
                                View Results
                              </Button>
                              <Button variant="outline" size="sm">
                                <Download className="h-4 w-4 mr-1" />
                                Report
                              </Button>
                            </>
                          )}
                          {job.status === "processing" && (
                            <Button variant="outline" size="sm" disabled>
                              Processing...
                            </Button>
                          )}
                          {job.status === "pending" && (
                            <Button variant="outline" size="sm" disabled>
                              In Queue
                            </Button>
                          )}
                        </div>
                      </div>
                    </Card>
                  </motion.div>
                ))}
              </div>
            )}
          </div>

          {jobs.length === 0 && !loading && (
            <Card className="p-12 text-center">
              <Clock className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
              <h3 className="text-lg font-semibold mb-2">No jobs yet</h3>
              <p className="text-muted-foreground mb-4">
                Submit your first prediction job to get started
              </p>
              <Button onClick={() => navigate("/predict")}>
                Create New Job
              </Button>
            </Card>
          )}
        </motion.div>
      </div>

      {showModal && selectedJob && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
          <div className="bg-white p-6 rounded shadow-lg max-w-lg w-full">
            <h2 className="text-xl font-bold mb-2">Result for {selectedJob.jobName || selectedJob.name || selectedJob.id || selectedJob.jobId}</h2>
            <pre className="bg-gray-100 p-2 rounded text-sm mb-4">{JSON.stringify(selectedJob.result || selectedJob.prediction || "No result available", null, 2)}</pre>
            <Button onClick={() => setShowModal(false)}>Close</Button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Jobs;
