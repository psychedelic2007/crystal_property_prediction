import React from 'react';
import Plot from 'react-plotly.js';

interface AttentionVisualizationProps {
  positions: number[][];
  attentionWeights?: number[];
  atomTypes?: number[];
  formula?: string;
}

const AttentionVisualization: React.FC<AttentionVisualizationProps> = ({
  positions,
  attentionWeights,
  atomTypes,
  formula
}) => {
  // Convert positions to arrays
  const x = positions.map(pos => pos[0]);
  const y = positions.map(pos => pos[1]);
  const z = positions.map(pos => pos[2]);

  // Normalize attention weights (same as your script)
  let colors: number[];
  if (attentionWeights && attentionWeights.length === positions.length) {
    const absWeights = attentionWeights.map(w => Math.abs(w));
    const maxWeight = Math.max(...absWeights);
    colors = absWeights.map(w => w / (maxWeight + 1e-8));
  } else {
    colors = new Array(positions.length).fill(0.5);
  }

  // Set marker sizes based on atom types (same as your script: atom_types*10)
  const markerSizes = atomTypes ? atomTypes.map(z => z * 10) : new Array(positions.length).fill(50);

  // Create hover text with atom information
  const hoverText = positions.map((pos, i) => {
    const atomType = atomTypes ? atomTypes[i] : 'Unknown';
    const attention = attentionWeights ? attentionWeights[i].toFixed(4) : 'N/A';
    return `Atom ${i}<br>Type: ${atomType}<br>Position: (${pos[0].toFixed(2)}, ${pos[1].toFixed(2)}, ${pos[2].toFixed(2)})<br>Attention: ${attention}`;
  });

  // 3D Structure Plot
  const plot3DData = [{
    x: x,
    y: y,
    z: z,
    mode: 'markers',
    type: 'scatter3d',
    marker: {
      size: markerSizes,
      color: colors,
      colorscale: 'Hot',
      colorbar: {
        title: 'Attention Weight',
        titleside: 'right'
      },
      line: {
        color: 'black',
        width: 1
      },
      opacity: 0.8
    },
    text: hoverText,
    hovertemplate: '%{text}<extra></extra>',
    name: 'Atoms'
  }];

  const plot3DLayout = {
    title: {
      text: `3D Structure with Attention${formula ? ` - ${formula}` : ''}`,
      font: { size: 16 }
    },
    scene: {
      xaxis: { title: 'X (Å)' },
      yaxis: { title: 'Y (Å)' },
      zaxis: { title: 'Z (Å)' },
      camera: {
        eye: { x: 1.5, y: 1.5, z: 1.5 }
      }
    },
    margin: { l: 0, r: 0, t: 50, b: 0 },
    showlegend: false
  };

  // XY Projection Plot
  const plotXYData = [{
    x: x,
    y: y,
    mode: 'markers',
    type: 'scatter',
    marker: {
      size: atomTypes ? atomTypes.map(z => z * 20) : new Array(positions.length).fill(100),
      color: colors,
      colorscale: 'Hot',
      colorbar: {
        title: 'Attention',
        titleside: 'right'
      },
      line: {
        color: 'black',
        width: 1
      },
      opacity: 0.8
    },
    text: hoverText,
    hovertemplate: '%{text}<extra></extra>',
    name: 'Atoms'
  }];

  const plotXYLayout = {
    title: {
      text: 'XY Projection',
      font: { size: 16 }
    },
    xaxis: { 
      title: 'X (Å)',
      scaleanchor: 'y',
      scaleratio: 1
    },
    yaxis: { title: 'Y (Å)' },
    margin: { l: 50, r: 30, t: 50, b: 50 },
    showlegend: false
  };

  return (
    <div className="space-y-6">
      {/* 3D Structure Plot */}
      <div className="bg-white rounded-lg border p-4">
        <Plot
          data={plot3DData}
          layout={plot3DLayout}
          config={{
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
            displaylogo: false
          }}
          style={{ width: '100%', height: '500px' }}
        />
      </div>

      {/* XY Projection Plot */}
      <div className="bg-white rounded-lg border p-4">
        <Plot
          data={plotXYData}
          layout={plotXYLayout}
          config={{
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
            displaylogo: false
          }}
          style={{ width: '100%', height: '400px' }}
        />
      </div>
    </div>
  );
};

export default AttentionVisualization;
