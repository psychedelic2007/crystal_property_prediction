# material_project_prediction.py - CORRECTED VERSION

import torch
from torch_geometric.data import Data, Batch
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.ext.matproj import MPRester
from pymatgen.core import Structure
from train_m3gnet_multitask import SimpleM3GNetInspired

import argparse, json, os
from itertools import combinations

# ------------------------------
# Load model configuration
# ------------------------------
def load_model_with_config(model_path, config_path, device):
    """Load model with correct architecture from config."""
    
    # Load hyperparameters
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    params = config['hyperparameters']
    
    print(f"Loading model with:")
    print(f"  Hidden dim: {params['hidden_dim']}")
    print(f"  Depth: {params['depth']}")
    print(f"  Dropout: {params['dropout']}")
    
    # Initialize with correct architecture
    model = SimpleM3GNetInspired(
        node_dim=4,
        edge_dim=3,
        space_group_dim=230,
        hidden=params['hidden_dim'],
        depth=params['depth'],
        dropout=params['dropout']
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model state dict (handles both formats)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Checkpoint saved with metadata
        state_dict = checkpoint['model_state_dict']
        print(f"  Loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Val MAE: {checkpoint.get('val_mae', 'N/A')}")
    else:
        # Checkpoint is already a state dict
        state_dict = checkpoint
    
    # Load weights
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully")
    
    return model

# ------------------------------
# Utility: Convert structure to PyG Data
# ------------------------------
def structure_to_pyg(structure, space_group=1):
    """
    Correct version - matches training preprocessing EXACTLY.
    Uses RAW features, not normalized!
    """
    from pymatgen.analysis.graphs import StructureGraph
    from pymatgen.analysis.local_env import CrystalNN
    from torch_geometric.utils import to_undirected
    
    # Build graph with CrystalNN (same as training)
    try:
        graph = StructureGraph.from_local_env_strategy(
            structure, 
            CrystalNN(x_diff_weight=0.0)
        )
    except:
        # Fallback for problematic structures
        return None
    
    edges = list(graph.graph.edges())
    if len(edges) == 0:
        return None
        
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_index = to_undirected(edge_index)
    
    # Node features - RAW values (no normalization!)
    node_features = []
    for site in structure:
        el = site.specie
        z = el.Z
        group = el.group
        row = el.row
        electroneg = el.X if (el.X and not np.isnan(el.X)) else 0.0
        
        node_features.append([z, group, row, electroneg])
    
    x = torch.tensor(node_features, dtype=torch.float32)
    
    # Edge features - RAW values (no normalization!)
    edge_attrs = []
    for idx in range(edge_index.size(1)):
        i, j = edge_index[0, idx].item(), edge_index[1, idx].item()
        dist = structure.get_distance(i, j)
        
        # Coordination counts (not normalized!)
        coord_i = len(structure.get_neighbors(structure[i], r=3.5))
        coord_j = len(structure.get_neighbors(structure[j], r=3.5))
        
        edge_attrs.append([dist, coord_i, coord_j])
    
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    
    # Positions
    pos = torch.tensor(structure.cart_coords, dtype=torch.float32)
    
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        space_group=torch.tensor([space_group - 1], dtype=torch.long),  # 0-indexed
        batch=torch.zeros(x.shape[0], dtype=torch.long)
    )
    
    return data

# ------------------------------
# Attention extractor helper
# ------------------------------
class AttentionExtractor:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.attention_weights = []
        self.node_attentions = []

        # Register hooks to capture attention layers
        for name, module in self.model.named_modules():
            if 'attention_pool' in name.lower():
                module.register_forward_hook(self.save_attention)
                print(f"  Hooked: {name}")

    def save_attention(self, module, input, output):
        if isinstance(output, tuple) and len(output) == 2:
            x, attn = output
            if attn is not None and isinstance(attn, torch.Tensor):
                self.node_attentions.append(attn.detach().cpu())

    def extract_attention_for_sample(self, data):
        self.attention_weights = []
        self.node_attentions = []
        
        # Add batch info
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=self.device)
        data = data.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(data)
        
        # Get node attention
        node_attention = None
        if len(self.node_attentions) > 0:
            node_attention = self.node_attentions[-1].squeeze().cpu().numpy()
        
        return node_attention, outputs

# ------------------------------
# Visualization
# ------------------------------
def visualize_attention(structure, attention_weights, predictions, save_path='attention_structure.png'):
    """Create comprehensive attention visualization."""
    
    fig = plt.figure(figsize=(16, 10))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    coords = np.array([s.coords for s in structure])
    atom_types = np.array([s.specie.Z for s in structure])
    
    # Normalize attention
    if attention_weights is not None and len(attention_weights) == len(coords):
        colors = np.abs(attention_weights)
        colors = colors / (colors.max() + 1e-8)
    else:
        colors = np.ones(len(coords)) * 0.5
    
    # Plot 1: 3D structure
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    sc = ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                    c=colors, cmap='hot', s=atom_types*10, 
                    alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('X (Å)')
    ax1.set_ylabel('Y (Å)')
    ax1.set_zlabel('Z (Å)')
    ax1.set_title('(a) 3D Structure with Attention')
    plt.colorbar(sc, ax=ax1, label='Attention Weight', shrink=0.6)
    
    # Plot 2: 2D projection
    ax2 = fig.add_subplot(gs[0, 1])
    sc2 = ax2.scatter(coords[:, 0], coords[:, 1], 
                     c=colors, cmap='hot', s=atom_types*20,
                     alpha=0.7, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('X (Å)')
    ax2.set_ylabel('Y (Å)')
    ax2.set_title('(b) XY Projection')
    ax2.set_aspect('equal')
    plt.colorbar(sc2, ax=ax2, label='Attention')
    
    # Plot 3: Attention distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(colors, bins=20, alpha=0.7, edgecolor='black')
    ax3.axvline(colors.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {colors.mean():.3f}')
    ax3.set_xlabel('Attention Weight')
    ax3.set_ylabel('Count')
    ax3.set_title('(c) Attention Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Top-5 atoms
    ax4 = fig.add_subplot(gs[1, 0])
    top_indices = np.argsort(colors)[-5:][::-1]
    
    from mendeleev import element
    try:
        labels = [f"Atom {i}\n{element(int(atom_types[i])).symbol}" for i in top_indices]
    except:
        labels = [f"Atom {i}\n(Z={int(atom_types[i])})" for i in top_indices]
    
    bars = ax4.barh(labels, colors[top_indices], color='red', alpha=0.7)
    ax4.set_xlabel('Attention Weight')
    ax4.set_title('(d) Top-5 Most Important Atoms')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Plot 5: Predictions
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis('off')
    
    crystal_systems = ['Triclinic', 'Monoclinic', 'Orthorhombic', 'Tetragonal', 
                      'Trigonal', 'Hexagonal', 'Cubic']
    material_types = ['Metal', 'Semiconductor', 'Insulator']
    stability_types = ['Stable', 'Metastable', 'Unstable']
    
    pred_text = f"""
PREDICTIONS:

Formula: {structure.composition.reduced_formula}
Num Atoms: {len(structure)}

Formation Energy: {predictions['energy'].item():.3f} eV/atom

Crystal System: {crystal_systems[predictions['crystal_system'].argmax().item()]}

Material Type: {material_types[predictions['material_type'].argmax().item()]}

Stability: {stability_types[predictions['stability'].argmax().item()]}

Space Group: {structure.get_space_group_info()[1]}
    """
    
    ax5.text(0.1, 0.9, pred_text, transform=ax5.transAxes,
            verticalalignment='top', fontfamily='monospace', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle(f'Materials Project: {structure.composition.reduced_formula}',
                fontsize=14, fontweight='bold')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved visualization to {save_path}")

# ------------------------------
# Main prediction + visualization
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description='Predict properties from Materials Project ID.')
    parser.add_argument('--mp_id', type=str, required=True, help='Materials Project ID (e.g., mp-149)')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model file')
    parser.add_argument('--config', type=str, required=True, help='Path to model config JSON')
    parser.add_argument('--mp_api_key', type=str, required=True, help='Materials Project API key')
    parser.add_argument('--output_dir', type=str, default='mp_predictions', help='Output directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*70)
    print(f"MATERIALS PROJECT PREDICTION: {args.mp_id}")
    print("="*70 + "\n")

    # Load structure from MP
    print("Fetching structure from Materials Project...")
    with MPRester(args.mp_api_key) as mpr:
        structure = mpr.get_structure_by_material_id(args.mp_id)
        
        # Try to get space group
        try:
            spacegroup_info = structure.get_space_group_info()
            space_group = spacegroup_info[1]  # Space group number
            print(f"  Formula: {structure.composition.reduced_formula}")
            print(f"  Space Group: {space_group} ({spacegroup_info[0]})")
        except:
            space_group = 1
            print(f"  Formula: {structure.composition.reduced_formula}")
            print(f"  Space Group: Unknown (using default)")

    # Convert to PyG Data
    print("\nConverting structure to graph...")
    data = structure_to_pyg(structure, space_group)
    print(f"  Nodes: {data.x.shape[0]}")
    print(f"  Edges: {data.edge_index.shape[1]}")

    # Load model
    print("\nLoading trained model...")
    model = load_model_with_config(args.model, args.config, device)
    print("✓ Model loaded successfully")

    # Extract attention and predict
    print("\nExtracting attention and making predictions...")
    extractor = AttentionExtractor(model, device)
    attention, predictions = extractor.extract_attention_for_sample(data)

    # Format results
    crystal_systems = ['Triclinic', 'Monoclinic', 'Orthorhombic', 'Tetragonal', 
                      'Trigonal', 'Hexagonal', 'Cubic']
    material_types = ['Metal', 'Semiconductor', 'Insulator']
    stability_types = ['Stable', 'Metastable', 'Unstable']
    
    results = {
        'mp_id': args.mp_id,
        'formula': structure.composition.reduced_formula,
        'space_group': int(space_group),
        'num_atoms': len(structure),
        'predictions': {
            'formation_energy_eV_per_atom': float(predictions['energy'].item()),
            'crystal_system': {
                'prediction': crystal_systems[predictions['crystal_system'].argmax().item()],
                'probabilities': predictions['crystal_system'].softmax(dim=0).tolist()
            },
            'material_type': {
                'prediction': material_types[predictions['material_type'].argmax().item()],
                'probabilities': predictions['material_type'].softmax(dim=0).tolist()
            },
            'stability': {
                'prediction': stability_types[predictions['stability'].argmax().item()],
                'probabilities': predictions['stability'].softmax(dim=0).tolist()
            }
        }
    }

    # Save results
    result_path = os.path.join(args.output_dir, f'{args.mp_id}_prediction.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved predictions to {result_path}")

    # Print summary
    print("\n" + "="*70)
    print("PREDICTION SUMMARY")
    print("="*70)
    print(f"Formation Energy: {results['predictions']['formation_energy_eV_per_atom']:.3f} eV/atom")
    print(f"Crystal System:   {results['predictions']['crystal_system']['prediction']}")
    print(f"Material Type:    {results['predictions']['material_type']['prediction']}")
    print(f"Stability:        {results['predictions']['stability']['prediction']}")

    # Visualize
    print("\nGenerating visualization...")
    viz_path = os.path.join(args.output_dir, f'{args.mp_id}_attention.png')
    visualize_attention(structure, attention, predictions, viz_path)

    print("\n" + "="*70)
    print("✅ PREDICTION COMPLETE!")
    print("="*70)

if __name__ == '__main__':
    main()
