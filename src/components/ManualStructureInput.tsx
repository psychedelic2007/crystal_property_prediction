import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";
import { Plus, Trash2 } from "lucide-react";
import { toast } from "sonner";

interface AtomSite {
  species: string;
  x: number;
  y: number;
  z: number;
}

interface StructureData {
  latticeVectors: number[][];
  atomSites: AtomSite[];
}

interface ManualStructureInputProps {
  onStructureSubmit: (data: StructureData) => void;
}

const ManualStructureInput = ({ onStructureSubmit }: ManualStructureInputProps) => {
  const [latticeVectors, setLatticeVectors] = useState<number[][]>([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ]);
  const [atomSites, setAtomSites] = useState<AtomSite[]>([
    { species: "Si", x: 0, y: 0, z: 0 },
  ]);

  const updateLatticeVector = (rowIdx: number, colIdx: number, value: string) => {
    const newVectors = [...latticeVectors];
    newVectors[rowIdx][colIdx] = parseFloat(value) || 0;
    setLatticeVectors(newVectors);
  };

  const addAtomSite = () => {
    setAtomSites([...atomSites, { species: "C", x: 0, y: 0, z: 0 }]);
  };

  const removeAtomSite = (index: number) => {
    if (atomSites.length === 1) {
      toast.error("At least one atom site is required");
      return;
    }
    setAtomSites(atomSites.filter((_, i) => i !== index));
  };

  const updateAtomSite = (index: number, field: keyof AtomSite, value: string | number) => {
    const newSites = [...atomSites];
    if (field === "species") {
      newSites[index][field] = value as string;
    } else {
      newSites[index][field] = parseFloat(value as string) || 0;
    }
    setAtomSites(newSites);
  };

  const handleSubmit = () => {
    // Basic validation
    if (atomSites.length === 0) {
      toast.error("Please add at least one atom site");
      return;
    }

    const hasEmptySpecies = atomSites.some(site => !site.species.trim());
    if (hasEmptySpecies) {
      toast.error("All atom sites must have a species defined");
      return;
    }

    onStructureSubmit({
      latticeVectors,
      atomSites,
    });
    
    toast.success("Structure data prepared for submission");
  };

  return (
    <div className="space-y-6">
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Lattice Vectors (Å)</h3>
        <div className="grid gap-4">
          {latticeVectors.map((vector, rowIdx) => (
            <div key={rowIdx} className="grid grid-cols-3 gap-3">
              {vector.map((value, colIdx) => (
                <div key={colIdx}>
                  <Label className="text-xs text-muted-foreground mb-1">
                    {['a', 'b', 'c'][rowIdx]}[{colIdx}]
                  </Label>
                  <Input
                    type="number"
                    step="0.01"
                    value={value}
                    onChange={(e) => updateLatticeVector(rowIdx, colIdx, e.target.value)}
                    className="text-sm"
                  />
                </div>
              ))}
            </div>
          ))}
        </div>
      </Card>

      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Atomic Sites (Fractional Coordinates)</h3>
          <Button onClick={addAtomSite} size="sm" variant="outline">
            <Plus className="h-4 w-4 mr-1" />
            Add Site
          </Button>
        </div>
        
        <div className="space-y-4">
          {atomSites.map((site, index) => (
            <div key={index} className="grid grid-cols-5 gap-3 items-end p-4 border rounded-lg">
              <div>
                <Label className="text-xs text-muted-foreground mb-1">Species</Label>
                <Input
                  value={site.species}
                  onChange={(e) => updateAtomSite(index, "species", e.target.value)}
                  placeholder="Si"
                  className="text-sm"
                />
              </div>
              <div>
                <Label className="text-xs text-muted-foreground mb-1">x</Label>
                <Input
                  type="number"
                  step="0.01"
                  value={site.x}
                  onChange={(e) => updateAtomSite(index, "x", e.target.value)}
                  className="text-sm"
                />
              </div>
              <div>
                <Label className="text-xs text-muted-foreground mb-1">y</Label>
                <Input
                  type="number"
                  step="0.01"
                  value={site.y}
                  onChange={(e) => updateAtomSite(index, "y", e.target.value)}
                  className="text-sm"
                />
              </div>
              <div>
                <Label className="text-xs text-muted-foreground mb-1">z</Label>
                <Input
                  type="number"
                  step="0.01"
                  value={site.z}
                  onChange={(e) => updateAtomSite(index, "z", e.target.value)}
                  className="text-sm"
                />
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => removeAtomSite(index)}
                className="text-destructive"
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          ))}
        </div>
      </Card>

      <Button onClick={handleSubmit} className="w-full" size="lg">
        Prepare Structure for Prediction
      </Button>
    </div>
  );
};

export default ManualStructureInput;
