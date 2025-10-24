"""
Process uploaded CIF files and extract features required for model inference.
"""

from pymatgen.core import Structure
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import torch
import numpy as np
from typing import Optional, Dict, Any


def get_node_features(structure: Structure) -> torch.Tensor:
    """Extract node features (atomic number, group, period, electronegativity)."""
    elements = [site.specie for site in structure]
    features = []
    for el in elements:
        electronegativity = el.X if el.X and not np.isnan(el.X) else 0.0
        features.append([el.Z, el.group, el.row, electronegativity])
    return torch.tensor(features, dtype=torch.float32)


def get_edge_features(structure: Structure, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Extract edge features ensuring they match edge_index.
    Features: [distance, coordination_number_i, coordination_number_j]
    """
    num_edges = edge_index.size(1)
    edge_attrs = []
    
    for idx in range(num_edges):
        i, j = edge_index[0, idx].item(), edge_index[1, idx].item()
        
        try:
            # Calculate distance between atoms
            dist = structure.get_distance(i, j)
            
            # Get coordination numbers
            coord_i = len(structure.get_neighbors(structure[i], r=3.5))
            coord_j = len(structure.get_neighbors(structure[j], r=3.5))
            
            edge_attrs.append([dist, coord_i, coord_j])
        except Exception:
            # Use default values if calculation fails
            edge_attrs.append([3.0, 6.0, 6.0])
    
    return torch.tensor(edge_attrs, dtype=torch.float32)


def get_atomic_f(structure: Structure) -> torch.Tensor:
    """Approximate atomic scattering factors (using atomic number as proxy)."""
    return torch.tensor([site.specie.Z for site in structure], dtype=torch.float32)


def structure_to_pyg(structure, space_group=1):
    """
    Convert structure to PyG Data - matches training preprocessing EXACTLY.
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


def process_cif_file(cif_path: str) -> Optional[Data]:
    """
    Process a CIF file and convert it to PyTorch Geometric Data format.
    Uses the exact same preprocessing as the user's trained model.
    
    Args:
        cif_path: Path to the CIF file
        
    Returns:
        Data object ready for model inference, or None if processing fails
    """
    try:
        # Load structure from CIF
        structure = Structure.from_file(cif_path)
        
        # Get space group number
        try:
            spacegroup_info = structure.get_space_group_info()
            space_group = spacegroup_info[1]  # Space group number
        except:
            space_group = 1
        
        # Use the exact same preprocessing as the user's model
        data = structure_to_pyg(structure, space_group)
        
        if data is None:
            raise ValueError("No bonds found in structure. This may be due to atoms being too far apart or an invalid structure. Please check your CIF file.")
        
        return data
        
    except Exception as e:
        print(f"Error processing CIF file: {e}")
        return None


def get_crystal_system(space_group: int) -> str:
    """Convert space group number to crystal system name."""
    if space_group < 2:
        return "Triclinic"
    elif space_group < 15:
        return "Monoclinic"
    elif space_group < 74:
        return "Orthorhombic"
    elif space_group < 142:
        return "Tetragonal"
    elif space_group < 167:
        return "Trigonal"
    elif space_group < 194:
        return "Hexagonal"
    else:
        return "Cubic"


def get_material_type(band_gap: float) -> str:
    """Classify material type based on band gap."""
    if band_gap < 0.1:
        return "Metal"
    elif band_gap < 2.0:
        return "Semiconductor"
    else:
        return "Insulator"


def get_stability_class(formation_energy: float) -> str:
    """Classify stability based on formation energy."""
    if formation_energy < -1.0:
        return "Stable"
    elif formation_energy < 0.0:
        return "Metastable"
    else:
        return "Unstable"


def extract_structure_info(cif_path: str) -> Dict[str, Any]:
    """
    Extract basic information about the structure for display.
    
    Args:
        cif_path: Path to the CIF file
        
    Returns:
        Dictionary with structure information
    """
    try:
        structure = Structure.from_file(cif_path)
        
        info = {
            'formula': structure.composition.reduced_formula,
            'num_atoms': len(structure),
            'num_elements': len(structure.composition.elements),
            'density': structure.density,
            'volume': structure.volume,
            'space_group_symbol': structure.get_space_group_info()[0],
            'space_group_number': structure.get_space_group_info()[1],
            'lattice_params': {
                'a': structure.lattice.a,
                'b': structure.lattice.b,
                'c': structure.lattice.c,
                'alpha': structure.lattice.alpha,
                'beta': structure.lattice.beta,
                'gamma': structure.lattice.gamma
            }
        }
        
        return info
        
    except Exception as e:
        return {'error': str(e)}


def predict_from_cif(model, cif_path: str, device: str = 'cpu') -> Optional[Dict[str, Any]]:
    """
    Complete pipeline: process CIF and make predictions.
    
    Args:
        model: Trained PyTorch model
        cif_path: Path to uploaded CIF file
        device: 'cpu' or 'cuda'
        
    Returns:
        Dictionary with predictions and structure info
    """
    # Extract structure information
    structure_info = extract_structure_info(cif_path)
    if 'error' in structure_info:
        return {'error': structure_info['error']}
    
    # Process CIF to Data object
    data = process_cif_file(cif_path)
    if data is None:
        return {'error': 'Failed to process CIF file'}
    
    # Move to device
    data = data.to(device)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(data)
        
        # Extract predictions (adjust based on your model's output format)
        # Assuming model returns: (formation_energy, band_gap, ...)
        formation_energy = predictions['formation_energy'].item()
        band_gap = predictions['band_gap'].item()
        space_group_pred = data.space_group.item() + 1  # Convert back to 1-indexed
    
    # Interpret predictions
    results = {
        'structure_info': structure_info,
        'predictions': {
            'formation_energy_per_atom': round(formation_energy, 4),
            'band_gap': round(band_gap, 4),
            'stability': get_stability_class(formation_energy),
            'material_type': get_material_type(band_gap),
            'crystal_system': get_crystal_system(space_group_pred - 1),
            'space_group_number': space_group_pred
        }
    }
    
    return results
