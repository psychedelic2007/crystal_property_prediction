import { useLocation, useNavigate } from "react-router-dom";
import Navigation from "@/components/Navigation";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import StructureViewer from "@/components/StructureViewer";
import { useEffect, useState } from "react";

const Result = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { result } = location.state || {};
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (result) setLoading(false);
  }, [result]);

  // Helper to render simple property list
  const renderProperties = (props: any) => {
    if (!props) return <div>No properties found.</div>;
    return (
      <div className="grid grid-cols-1 gap-2">
        {Object.entries(props).map(([k, v]) => (
          <div key={k} className="flex justify-between border-b py-1">
            <div className="font-medium">{k}</div>
            <div className="text-right">{String(v)}</div>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      <div className="container mx-auto px-4 py-12">
        <Card className="max-w-5xl mx-auto p-8">
          <h1 className="text-3xl font-bold mb-4">Prediction Results</h1>

          {loading ? (
            <div className="flex flex-col items-center justify-center py-12">
              <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-primary mb-4"></div>
              <div className="text-lg">Making predictions...</div>
            </div>
          ) : result ? (
            <div className="grid lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <h2 className="text-xl font-semibold mb-2">
                  Material: {result.material || result.name || "-"}
                </h2>
                <div className="mb-4">
                  Status:{" "}
                  <span className="font-bold text-green-600">Completed</span>
                </div>

                <div className="mb-6">
                  <h3 className="font-semibold mb-2">Predicted Properties</h3>
                  <div className="grid grid-cols-1 gap-4">
                    {result.prediction && (
                      <>
                        <div className="bg-blue-50 p-4 rounded-lg border">
                          <div className="flex justify-between items-center">
                            <span className="font-medium text-blue-900">Formation Energy</span>
                            <span className="text-2xl font-bold text-blue-600">
                              {result.prediction.formation_energy_per_atom?.toFixed(4) || 'N/A'} eV/atom
                            </span>
                          </div>
                        </div>
                        
                        {result.prediction.stability && (
                          <div className="bg-green-50 p-4 rounded-lg border">
                            <div className="flex justify-between items-center">
                              <span className="font-medium text-green-900">Stability</span>
                              <span className="text-lg font-semibold text-green-600">
                                {result.prediction.stability}
                              </span>
                            </div>
                          </div>
                        )}
                        
                        {result.prediction.crystal_system && (
                          <div className="bg-purple-50 p-4 rounded-lg border">
                            <div className="flex justify-between items-center">
                              <span className="font-medium text-purple-900">Crystal System</span>
                              <span className="text-lg font-semibold text-purple-600">
                                {result.prediction.crystal_system}
                              </span>
                            </div>
                          </div>
                        )}
                        
                        {result.prediction.space_group_number && (
                          <div className="bg-gray-50 p-4 rounded-lg border">
                            <div className="flex justify-between items-center">
                              <span className="font-medium text-gray-900">Space Group</span>
                              <span className="text-lg font-semibold text-gray-600">
                                {result.prediction.space_group_number}
                              </span>
                            </div>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </div>

                <div className="mb-6">
                  <h3 className="font-semibold mb-2">Additional Info</h3>
                  <pre className="bg-gray-100 p-3 rounded text-sm max-h-48 overflow-auto">
                    {JSON.stringify(result.info || {}, null, 2)}
                  </pre>
                </div>

                {/* If plot images or data provided by backend, show here */}
                {result.plots && Array.isArray(result.plots) && (
                  <div className="mb-4">
                    <h3 className="font-semibold mb-2">Plots</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {result.plots.map((p: any, idx: number) => (
                        // backend can supply either image data URI or URL
                        <div key={idx} className="border p-2 rounded">
                          {typeof p === "string" ? (
                            <img
                              src={p}
                              alt={`plot-${idx}`}
                              className="w-full h-auto"
                            />
                          ) : (
                            <pre className="text-xs">
                              {JSON.stringify(p, null, 2)}
                            </pre>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <div className="lg:col-span-1">
                <h3 className="text-lg font-semibold mb-3">Structure Preview</h3>
                <div
                  className="border rounded p-2 mb-4"
                  style={{ height: 420 }}
                >
                  {/* Use existing StructureViewer component; pass attention if available */}
                  <StructureViewer
                    hasStructure={!!result.structure_info}
                    // @ts-ignore - dynamic prop for attention visualization
                    structureInfo={result.structure_info}
                    // @ts-ignore
                    attention={result.attention}
                  />
                </div>

                {result.metrics && (
                  <div className="mb-4">
                    <h4 className="font-semibold mb-2">Metrics</h4>
                    {renderProperties(result.metrics)}
                  </div>
                )}

                <div className="mt-4">
                  <Button onClick={() => navigate("/")}>Back</Button>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-12">
              <h2 className="text-2xl font-bold mb-2">404</h2>
              <div className="mb-4">Oops! Page not found</div>
              <Button onClick={() => navigate("/")}>Return to Home</Button>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
};

export default Result;
