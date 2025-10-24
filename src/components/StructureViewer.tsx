import { useEffect, useRef } from "react";
import { Box, Atom, Zap } from "lucide-react";

interface StructureInfo {
  formula: string;
  num_atoms: number;
  num_elements: number;
  density: number;
  volume: number;
  space_group_symbol: string;
  space_group_number: number;
  lattice_params: {
    a: number;
    b: number;
    c: number;
    alpha: number;
    beta: number;
    gamma: number;
  };
}

interface StructureViewerProps {
  hasStructure: boolean;
  structureInfo?: StructureInfo;
}

const StructureViewer = ({ hasStructure, structureInfo }: StructureViewerProps) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!hasStructure || !containerRef.current) return;

    // In production, integrate 3Dmol.js here
    // For now, show a placeholder
    
  }, [hasStructure]);

  if (!hasStructure) {
    return (
      <div className="aspect-square bg-muted/30 rounded-lg flex flex-col items-center justify-center text-muted-foreground">
        <Box className="h-12 w-12 mb-3 opacity-50" />
        <p className="text-sm">Structure will appear here</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* 3D Viewer */}
      <div 
        ref={containerRef}
        className="aspect-square bg-gradient-to-br from-primary/5 to-accent/5 rounded-lg flex items-center justify-center relative overflow-hidden"
      >
        {/* Placeholder for 3D viewer */}
        <div className="text-center p-6">
          <Atom className="h-16 w-16 text-primary mx-auto mb-3 animate-pulse" />
          <p className="text-sm font-medium">3D Structure Loaded</p>
          <p className="text-xs text-muted-foreground mt-1">
            Interactive viewer will render here
          </p>
        </div>
        
        {/* Note: Integrate 3Dmol.js or NGL viewer here */}
        {/* Example: <div id="3dmol-container" style={{ width: '100%', height: '100%' }}></div> */}
      </div>

      {/* Structure Information */}
      {structureInfo && (
        <div className="bg-muted/30 rounded-lg p-4 space-y-3">
          <div className="flex items-center gap-2 mb-3">
            <Zap className="h-4 w-4 text-primary" />
            <h4 className="font-semibold text-sm">Structure Details</h4>
          </div>
          
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-muted-foreground">Formula:</span>
              <span className="ml-1 font-medium">{structureInfo.formula}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Atoms:</span>
              <span className="ml-1 font-medium">{structureInfo.num_atoms}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Elements:</span>
              <span className="ml-1 font-medium">{structureInfo.num_elements}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Space Group:</span>
              <span className="ml-1 font-medium">{structureInfo.space_group_symbol}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Density:</span>
              <span className="ml-1 font-medium">{structureInfo.density.toFixed(2)} g/cm³</span>
            </div>
            <div>
              <span className="text-muted-foreground">Volume:</span>
              <span className="ml-1 font-medium">{structureInfo.volume.toFixed(1)} Å³</span>
            </div>
          </div>
          
          <div className="pt-2 border-t border-border/50">
            <div className="text-xs text-muted-foreground">
              <div>Lattice: a={structureInfo.lattice_params.a.toFixed(2)}Å, 
                   b={structureInfo.lattice_params.b.toFixed(2)}Å, 
                   c={structureInfo.lattice_params.c.toFixed(2)}Å</div>
              <div>Angles: α={structureInfo.lattice_params.alpha.toFixed(1)}°, 
                   β={structureInfo.lattice_params.beta.toFixed(1)}°, 
                   γ={structureInfo.lattice_params.gamma.toFixed(1)}°</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default StructureViewer;
