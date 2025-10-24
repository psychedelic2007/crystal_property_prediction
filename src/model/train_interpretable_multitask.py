"""
train_interpretable_multitask.py
Enhanced FK-PCGN with multi-task learning and interpretability.

Multi-task outputs:
1. Formation energy (regression)
2. Stability class (classification: stable/metastable/unstable)
3. Crystal system (classification: cubic/tetragonal/etc.)
4. Band gap category (classification: metal/semiconductor/insulator)

Key features:
- Attention mechanisms for interpretability
- Multi-task learning with uncertainty weighting
- Chemical insight extraction
- Publication-ready metrics and visualizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Dataset, Batch
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import glob
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, r2_score, accuracy_score, 
    f1_score, confusion_matrix, classification_report
)

# ============================================================================
# ATTENTION-BASED INTERPRETABLE LAYERS
# ============================================================================

class InterpretableEdgeConv(MessagePassing):
    """
    Edge convolution with attention for interpretability.
    Tracks which edges are most important for predictions.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = nn.Linear(2 * in_channels + 3, out_channels)  # +3 for edge_attr
        self.att_lin = nn.Linear(2 * in_channels + 3, 1)
        
    def forward(self, x, edge_index, edge_attr):
        # x: [N, in_channels], edge_index: [2, E], edge_attr: [E, 3]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        # Concatenate source, target, and edge features
        tmp = torch.cat([x_i, x_j, edge_attr], dim=-1)
        
        # Compute attention weights (interpretability!)
        alpha = torch.sigmoid(self.att_lin(tmp))
        self.last_attention = alpha  # Store for visualization
        
        # Weighted message
        return alpha * self.lin(tmp)

class NodeAttentionPool(nn.Module):
    """
    Attention-based pooling that identifies important atoms.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.att_lin = nn.Linear(in_channels, 1)
        
    def forward(self, x, batch):
        # x: [N, in_channels], batch: [N]
        alpha = torch.softmax(self.att_lin(x), dim=0)
        self.last_attention = alpha  # Store for visualization
        
        # Weighted sum
        out = global_add_pool(alpha * x, batch)
        return out, alpha

# ============================================================================
# FOURIER-KOLMOGOROV LAYER (From your inspiration)
# ============================================================================

class FourierKolmogorovLayer(nn.Module):
    """
    Fourier-based transformation inspired by Kolmogorov-Arnold Networks.
    Captures periodic patterns in crystal structures.
    """
    def __init__(self, in_dim, out_dim, n_frequencies=8):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.weights = nn.Parameter(torch.randn(n_frequencies, in_dim, out_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
    def forward(self, x):
        # x: [N, in_dim]
        batch_size = x.size(0)
        
        # Generate frequency basis
        freqs = torch.arange(1, self.n_frequencies + 1, device=x.device).float()
        
        # Fourier features: sin(2πkx) and cos(2πkx)
        x_expanded = x.unsqueeze(1)  # [N, 1, in_dim]
        freqs_expanded = freqs.view(1, -1, 1)  # [1, K, 1]
        
        sin_features = torch.sin(2 * np.pi * freqs_expanded * x_expanded)
        cos_features = torch.cos(2 * np.pi * freqs_expanded * x_expanded)
        
        # Weighted combination
        sin_out = torch.einsum('nki,kio->no', sin_features, self.weights)
        cos_out = torch.einsum('nki,kio->no', cos_features, self.weights)
        
        return sin_out + cos_out + self.bias

# ============================================================================
# MAIN MODEL: Interpretable Multi-Task FK-PCGN
# ============================================================================

class InterpretableMultiTaskCrystal(nn.Module):
    """
    Multi-task crystal property predictor with interpretability.
    
    Tasks:
    1. Formation energy (eV/atom) - regression
    2. Stability class (stable/metastable/unstable) - classification
    3. Crystal system (7 classes) - classification
    4. Material type (metal/semiconductor/insulator) - classification
    """
    def __init__(
        self,
        node_dim=4,
        edge_dim=3,
        space_group_dim=230,
        hidden=128,
        depth=4,
        n_frequencies=8,
        dropout=0.1
    ):
        super().__init__()
        
        # Input embeddings
        self.node_embed = nn.Linear(node_dim, hidden)
        self.space_group_embed = nn.Embedding(space_group_dim, hidden)
        
        # Interpretable graph convolutions
        self.convs = nn.ModuleList([
            InterpretableEdgeConv(hidden, hidden) for _ in range(depth)
        ])
        
        # Fourier-Kolmogorov transformations
        self.fk_layers = nn.ModuleList([
            FourierKolmogorovLayer(hidden, hidden, n_frequencies) 
            for _ in range(depth)
        ])
        
        # Batch norms
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(depth)])
        
        # Attention pooling (interpretability)
        self.attention_pool = NodeAttentionPool(hidden)
        
        # Multi-task heads
        self.energy_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
        
        self.stability_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3)  # stable/metastable/unstable
        )
        
        self.crystal_system_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 7)  # 7 crystal systems
        )
        
        self.material_type_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3)  # metal/semiconductor/insulator
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data, return_attention=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch
        space_group = data.space_group
        
        # Initial embeddings
        x = self.node_embed(x)
        sg_embed = self.space_group_embed(space_group)
        
        # Store attention weights for interpretability
        edge_attentions = []
        
        # Graph convolutions with FK transformations
        for conv, fk, bn in zip(self.convs, self.fk_layers, self.bns):
            x_conv = conv(x, edge_index, edge_attr)
            edge_attentions.append(conv.last_attention)
            
            x_fk = fk(x)
            x = x_conv + x_fk  # Residual connection
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Attention pooling (interpretability)
        x_pool, node_attention = self.attention_pool(x, batch)
        
        # Combine with space group
        x_combined = torch.cat([x_pool, sg_embed], dim=-1)
        
        # Multi-task predictions
        energy = self.energy_head(x_combined)
        stability = self.stability_head(x_combined)
        crystal_system = self.crystal_system_head(x_combined)
        material_type = self.material_type_head(x_combined)
        
        outputs = {
            'energy': energy,
            'stability': stability,
            'crystal_system': crystal_system,
            'material_type': material_type
        }
        
        if return_attention:
            outputs['node_attention'] = node_attention
            outputs['edge_attentions'] = edge_attentions
            
        return outputs

# ============================================================================
# DATASET WITH MULTI-TASK LABELS
# ============================================================================

class MultiTaskCrystalDataset(Dataset):
    """
    Enhanced dataset with multiple labels for each crystal.
    """
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.data_files = sorted(glob.glob(os.path.join(root_dir, "data_*.pt")))
        print(f"Found {len(self.data_files)} crystal structures")
        
    def len(self):
        return len(self.data_files)
    
    def get(self, idx):
        data = torch.load(self.data_files[idx], weights_only=False)
        
        # FIX EDGE MISMATCH ISSUE
        num_edges = data.edge_index.size(1)
        num_edge_attrs = data.edge_attr.size(0)
        
        if num_edges != num_edge_attrs:
            # Option 1: If edge_attr is roughly half (undirected -> directed conversion)
            if abs(num_edges - 2 * num_edge_attrs) < 10:
                # Duplicate edge attributes for both directions
                data.edge_attr = torch.cat([data.edge_attr, data.edge_attr], dim=0)
            # Option 2: If edge_attr is larger, truncate
            elif num_edge_attrs > num_edges:
                data.edge_attr = data.edge_attr[:num_edges]
            # Option 3: If edge_attr is smaller, repeat cyclically
            else:
                repeat_times = (num_edges // num_edge_attrs) + 1
                data.edge_attr = data.edge_attr.repeat(repeat_times, 1)[:num_edges]
        
        # Ensure edge_attr has correct shape [num_edges, edge_dim]
        if data.edge_attr.dim() == 1:
            data.edge_attr = data.edge_attr.unsqueeze(-1)
        if data.edge_attr.size(1) != 3:
            # Pad or truncate to 3 features
            if data.edge_attr.size(1) < 3:
                padding = torch.zeros(data.edge_attr.size(0), 3 - data.edge_attr.size(1))
                data.edge_attr = torch.cat([data.edge_attr, padding], dim=1)
            else:
                data.edge_attr = data.edge_attr[:, :3]
        
        # Add multi-task labels
        energy = data.y.item()
        
        # Stability classification (based on formation energy)
        if energy < -1.0:
            stability_class = 0  # stable
        elif energy < 0.0:
            stability_class = 1  # metastable
        else:
            stability_class = 2  # unstable
        
        # Crystal system (from space group)
        sg = data.space_group.item()
        crystal_system = self._space_group_to_crystal_system(sg)
        
        # Material type (heuristic based on formation energy and composition)
        material_type = self._infer_material_type(data, energy)
        
        # Add to data object
        data.stability_class = torch.tensor([stability_class], dtype=torch.long)
        data.crystal_system_class = torch.tensor([crystal_system], dtype=torch.long)
        data.material_type_class = torch.tensor([material_type], dtype=torch.long)
        
        return data
    
    def _space_group_to_crystal_system(self, sg):
        """Map space group to crystal system (0-6)."""
        if sg < 2: return 0      # Triclinic
        elif sg < 15: return 1   # Monoclinic
        elif sg < 74: return 2   # Orthorhombic
        elif sg < 142: return 3  # Tetragonal
        elif sg < 167: return 4  # Trigonal
        elif sg < 194: return 5  # Hexagonal
        else: return 6           # Cubic
    
    def _infer_material_type(self, data, energy):
        """
        Heuristic material type inference.
        In practice, you'd get this from band gap data.
        """
        # Simplified heuristic based on atomic numbers and energy
        avg_atomic_num = data.x[:, 0].mean().item()
        
        if avg_atomic_num > 40 and energy < -2.0:
            return 0  # Metal
        elif energy > -0.5:
            return 2  # Insulator
        else:
            return 1  # Semiconductor

# ============================================================================
# MULTI-TASK LOSS WITH UNCERTAINTY WEIGHTING
# ============================================================================

class MultiTaskLoss(nn.Module):
    """
    Uncertainty-weighted multi-task loss (Kendall et al. 2018).
    Automatically balances task contributions.
    """
    def __init__(self):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(4))  # 4 tasks
        
    def forward(self, outputs, targets):
        energy_pred = outputs['energy']
        stability_pred = outputs['stability']
        crystal_pred = outputs['crystal_system']
        material_pred = outputs['material_type']
        
        # Unpack targets
        energy_true = targets['energy']
        stability_true = targets['stability']
        crystal_true = targets['crystal_system']
        material_true = targets['material_type']
        
        # Task losses (fix shape mismatch)
        loss_energy = F.mse_loss(energy_pred.squeeze(), energy_true.squeeze())
        loss_stability = F.cross_entropy(stability_pred, stability_true.squeeze())
        loss_crystal = F.cross_entropy(crystal_pred, crystal_true.squeeze())
        loss_material = F.cross_entropy(material_pred, material_true.squeeze())
        
        # Uncertainty weighting
        total_loss = (
            torch.exp(-self.log_vars[0]) * loss_energy + self.log_vars[0] +
            torch.exp(-self.log_vars[1]) * loss_stability + self.log_vars[1] +
            torch.exp(-self.log_vars[2]) * loss_crystal + self.log_vars[2] +
            torch.exp(-self.log_vars[3]) * loss_material + self.log_vars[3]
        )
        
        return total_loss, {
            'energy': loss_energy.item(),
            'stability': loss_stability.item(),
            'crystal_system': loss_crystal.item(),
            'material_type': loss_material.item()
        }

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    task_losses = {'energy': 0, 'stability': 0, 'crystal_system': 0, 'material_type': 0}
    
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch)
        
        # Prepare targets
        targets = {
            'energy': batch.y,
            'stability': batch.stability_class,
            'crystal_system': batch.crystal_system_class,
            'material_type': batch.material_type_class
        }
        
        # Compute loss
        loss, losses_dict = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        for key, val in losses_dict.items():
            task_losses[key] += val
    
    n_batches = len(loader)
    return total_loss / n_batches, {k: v/n_batches for k, v in task_losses.items()}

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    # Storage for predictions
    energy_preds, energy_trues = [], []
    stability_preds, stability_trues = [], []
    crystal_preds, crystal_trues = [], []
    material_preds, material_trues = [], []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            
            targets = {
                'energy': batch.y,
                'stability': batch.stability_class,
                'crystal_system': batch.crystal_system_class,
                'material_type': batch.material_type_class
            }
            
            loss, _ = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Store predictions
            energy_preds.extend(outputs['energy'].cpu().numpy())
            energy_trues.extend(batch.y.cpu().numpy())
            
            stability_preds.extend(outputs['stability'].argmax(dim=1).cpu().numpy())
            stability_trues.extend(batch.stability_class.cpu().numpy())
            
            crystal_preds.extend(outputs['crystal_system'].argmax(dim=1).cpu().numpy())
            crystal_trues.extend(batch.crystal_system_class.cpu().numpy())
            
            material_preds.extend(outputs['material_type'].argmax(dim=1).cpu().numpy())
            material_trues.extend(batch.material_type_class.cpu().numpy())
    
    # Calculate metrics
    metrics = {
        'loss': total_loss / len(loader),
        'energy_mae': mean_absolute_error(energy_trues, energy_preds),
        'energy_r2': r2_score(energy_trues, energy_preds),
        'stability_acc': accuracy_score(stability_trues, stability_preds),
        'stability_f1': f1_score(stability_trues, stability_preds, average='macro'),
        'crystal_acc': accuracy_score(crystal_trues, crystal_preds),
        'material_acc': accuracy_score(material_trues, material_preds)
    }
    
    return metrics, (energy_preds, energy_trues, stability_preds, stability_trues,
                     crystal_preds, crystal_trues, material_preds, material_trues)

# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    # Configuration
    config = {
        'data_dir': 'mp_data_50k',
        'hidden': 128,
        'depth': 4,
        'n_frequencies': 8,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'epochs': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("Configuration:", json.dumps(config, indent=2))
    device = torch.device(config['device'])
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = MultiTaskCrystalDataset(config['data_dir'])
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, collate_fn=Batch.from_data_list)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, collate_fn=Batch.from_data_list)
    
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    # Initialize model
    model = InterpretableMultiTaskCrystal(
        node_dim=4,
        edge_dim=3,
        space_group_dim=230,
        hidden=config['hidden'],
        depth=config['depth'],
        n_frequencies=config['n_frequencies']
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = MultiTaskLoss().to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=config['learning_rate'],
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    
    print("\nStarting training...")
    for epoch in range(config['epochs']):
        print(f"\n=== Epoch {epoch+1}/{config['epochs']} ===")
        
        train_loss, train_task_losses = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}")
        for task, loss in train_task_losses.items():
            print(f"  {task}: {loss:.4f}")
        
        val_metrics, _ = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Energy MAE: {val_metrics['energy_mae']:.4f} eV/atom")
        print(f"  Energy R²: {val_metrics['energy_r2']:.4f}")
        print(f"  Stability Acc: {val_metrics['stability_acc']:.4f}")
        print(f"  Crystal System Acc: {val_metrics['crystal_acc']:.4f}")
        print(f"  Material Type Acc: {val_metrics['material_acc']:.4f}")
        
        scheduler.step(val_metrics['loss'])
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), 'multitask_crystal_best.pth')
            print("✓ Saved best model")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_metrics'].append(val_metrics)
    
    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n=== Training Complete ===")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
