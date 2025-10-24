import { motion } from "framer-motion";
import { useParams } from "react-router-dom";
import { useEffect, useState } from "react";
import { apiClient, PredictionResult } from "@/lib/api-client";
import Navigation from "@/components/Navigation";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Download, Share2 } from "lucide-react";
import Plot from "react-plotly.js";
import AttentionVisualization from "@/components/AttentionVisualization";

const Results = () => {
  const { jobId } = useParams();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;

    const poll = async () => {
      try {
        // first check status
        const status = await apiClient.getJobStatus(jobId);
        if (status.status === 'completed') {
          const res = await apiClient.getResults(jobId);
          if (!cancelled) {
            setResult(res);
            setLoading(false);
          }
        } else if (status.status === 'failed') {
          if (!cancelled) {
            setError(status.message || 'Job failed');
            setLoading(false);
          }
        } else {
          // keep polling
          setTimeout(poll, 1500);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : 'Failed to fetch results');
          setLoading(false);
        }
      }
    };

    poll();
    return () => { cancelled = true; };
  }, [jobId]);

  // Mock plot data for parity plot
  const parityData = [
    {
      x: [1.2, 2.3, 3.1, 4.5, 5.2, 6.1],
      y: [1.3, 2.2, 3.3, 4.4, 5.1, 6.2],
      mode: 'markers',
      type: 'scatter',
      name: 'Predictions',
      marker: { color: '#3b82f6', size: 10 },
    },
    {
      x: [1, 6.5],
      y: [1, 6.5],
      mode: 'lines',
      type: 'scatter',
      name: 'Ideal',
      line: { color: 'gray', dash: 'dash' },
    },
  ];

  // Mock feature importance data
  const featureImportance = [
    { feature: 'Atomic Radius', importance: 0.35 },
    { feature: 'Electronegativity', importance: 0.28 },
    { feature: 'Valence Electrons', importance: 0.20 },
    { feature: 'Crystal System', importance: 0.12 },
    { feature: 'Coordination Number', importance: 0.05 },
  ];

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <div className="container mx-auto px-4 py-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-6xl mx-auto"
        >
          {/* Header */}
          <div className="mb-8 flex items-start justify-between flex-wrap gap-4">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <h1 className="text-4xl font-bold">{result?.structure_info?.formula || 'Prediction Results'}</h1>
                {!loading && !error && (
                  <Badge className="bg-green-500/10 text-green-700">Completed</Badge>
                )}
              </div>
              <p className="text-muted-foreground">Job ID: {jobId}</p>
            </div>
            <div className="flex gap-2">
              <Button variant="outline">
                <Share2 className="h-4 w-4 mr-2" />
                Share
              </Button>
              <Button className="gradient-primary">
                <Download className="h-4 w-4 mr-2" />
                Download Report
              </Button>
            </div>
          </div>

          {/* Loading / Error */}
          {loading && (
            <Card className="p-6 mb-6">
              <p className="text-sm text-muted-foreground">Fetching results...</p>
            </Card>
          )}
          {error && (
            <Card className="p-6 mb-6">
              <p className="text-sm text-red-600">{error}</p>
            </Card>
          )}

          {/* Main Results */}
          <div className="grid md:grid-cols-2 gap-6 mb-6">
            <Card className="p-6">
              <h2 className="text-lg font-semibold mb-4">Prediction Results</h2>
              <div className="space-y-4">
                <div>
                  <p className="text-sm text-muted-foreground mb-1">Property</p>
                  <p className="text-2xl font-bold">Formation Energy (eV/atom)</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground mb-1">Predicted Value</p>
                  <p className="text-4xl font-bold text-primary">
                    {result?.prediction?.formation_energy_per_atom?.toFixed(4)} eV/atom
                  </p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground mb-1">Stability</p>
                  <p className="text-lg font-semibold text-green-600">
                    {result?.prediction?.stability || 'N/A'}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground mb-1">Crystal System</p>
                  <p className="text-lg font-semibold text-purple-600">
                    {result?.prediction?.crystal_system || 'N/A'}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground mb-1">Model Version</p>
                  <p className="font-medium">{result?.modelVersion || '1.0.0'}</p>
                </div>
              </div>
            </Card>

            <Card className="p-6">
              <h2 className="text-lg font-semibold mb-4">Model Interpretation</h2>
              <p className="text-sm text-muted-foreground mb-4">
                The formation energy prediction is based on structural features extracted from your crystal structure. 
                The attention visualization shows which atoms contribute most to the prediction.
              </p>
              <div className="p-4 bg-muted/30 rounded-lg">
                <p className="text-sm font-medium mb-2">Key Insights:</p>
                <ul className="text-sm space-y-1 text-muted-foreground">
                  <li>• Formation energy indicates thermodynamic stability</li>
                  <li>• Negative values suggest stable compound formation</li>
                  <li>• Attention weights highlight critical atomic environments</li>
                </ul>
              </div>
            </Card>
          </div>

          {/* Interactive Attention Visualizations */}
          <Card className="p-6">
            <h2 className="text-lg font-semibold mb-4">Interactive Attention Visualization</h2>
            {(() => {
              const positions = (result as any)?.positions;
              const attentionWeights = (result as any)?.attention?.node as number[] | undefined;
              const atomTypes = (result as any)?.atom_types as number[] | undefined;
              const formula = result?.structure_info?.formula;
              
              // Debug logging
              console.log('Results data:', {
                positions: positions?.length,
                attentionWeights: attentionWeights?.length,
                atomTypes: atomTypes?.length,
                formula,
                fullResult: result
              });
              
              if (!positions || positions.length === 0) {
                return (
                  <div className="text-center py-8 text-muted-foreground">
                    No structure data available for visualization
                    <div className="text-xs mt-2">
                      Positions: {positions ? positions.length : 'undefined'}
                    </div>
                  </div>
                );
              }

              // Use atom types from backend, fallback to default if not available
              const finalAtomTypes = atomTypes || new Array(positions.length).fill(6);
              
              return (
                <AttentionVisualization
                  positions={positions}
                  attentionWeights={attentionWeights}
                  atomTypes={finalAtomTypes}
                  formula={formula}
                />
              );
            })()}
          </Card>

          {/* Additional Analysis Plots */}
          <div className="grid lg:grid-cols-2 gap-6">
            <Card className="p-6">
              <h2 className="text-lg font-semibold mb-4">Attention Distribution</h2>
              <Plot
                data={(() => {
                  const attn = (result as any)?.attention?.node as number[] | undefined;
                  if (!attn || attn.length === 0) return [] as any;
                  return [{ x: attn, type: 'histogram', marker: { color: '#ef4444' } }];
                })() as any}
                layout={{
                  width: undefined,
                  height: 300,
                  autosize: true,
                  xaxis: { title: 'Attention Weight' },
                  yaxis: { title: 'Count' },
                  showlegend: false,
                  margin: { l: 50, r: 30, t: 30, b: 50 },
                }}
                config={{ responsive: true, displayModeBar: false }}
                className="w-full"
              />
            </Card>

            <Card className="p-6">
              <h2 className="text-lg font-semibold mb-4">Top-5 Most Important Atoms</h2>
              <Plot
                data={(() => {
                  const attn = (result as any)?.attention?.node as number[] | undefined;
                  if (!attn || attn.length === 0) return [] as any;
                  const indices = attn.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 5);
                  const y = indices.map(pair => `Atom ${pair[1]}`);
                  const x = indices.map(pair => pair[0]);
                  return [{ x, y, type: 'bar', orientation: 'h', marker: { color: '#ef4444' } }];
                })() as any}
                layout={{
                  width: undefined,
                  height: 300,
                  autosize: true,
                  xaxis: { title: 'Attention' },
                  margin: { l: 100, r: 30, t: 30, b: 50 },
                }}
                config={{ responsive: true, displayModeBar: false }}
                className="w-full"
              />
            </Card>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Results;
