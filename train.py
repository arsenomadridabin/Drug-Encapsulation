import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import json
from datetime import datetime

from parse_itp import parse_nbfix_table
from build_graphs import MolecularGraphBuilder
from gnn_model import EncapsulationGNN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1.0, neginf=0.0)
    
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    try:
        spearman_corr, _ = spearmanr(y_true, y_pred)
        spearman_corr = 0.0 if np.isnan(spearman_corr) else spearman_corr
    except:
        spearman_corr = 0.0
    
    return {'mae': float(mae), 'mse': float(mse), 'r2': float(r2), 'spearman': float(spearman_corr)}


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                   batch.bead_type_id, batch.num_atoms, batch.num_bonds,
                   batch.avg_degree, batch.max_degree, batch.graph_density,
                   batch.total_charge, batch.charge_std, batch.unique_bead_types)
        
        loss = criterion(out.squeeze(), batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                       batch.bead_type_id, batch.num_atoms, batch.num_bonds,
                       batch.avg_degree, batch.max_degree, batch.graph_density,
                       batch.total_charge, batch.charge_std, batch.unique_bead_types)
            all_preds.append(out.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    return compute_metrics(all_targets, all_preds), all_preds, all_targets


def train_model(model, train_loader, val_loader, device, config):
    def weighted_mse_loss(pred, target):
        base_weights = 1.0 + 2.5 * target
        error_magnitude = torch.abs(pred - target)
        error_penalty = 1.0 + 0.5 * torch.clamp(error_magnitude - 0.2, min=0.0)
        weights = base_weights * error_penalty
        return torch.mean(weights * (pred - target) ** 2)
    
    criterion = weighted_mse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_mae = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(config['max_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _, _ = validate(model, val_loader, device)
        scheduler.step(val_metrics['mae'])
        
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            if epoch >= 50:
                patience_counter += 1
                if patience_counter >= config['early_stopping_patience']:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
            else:
                patience_counter = 0
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val MAE={val_metrics['mae']:.4f}, "
                  f"Val R²={val_metrics['r2']:.4f}, Val Spearman={val_metrics['spearman']:.4f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return validate(model, val_loader, device)


def main():
    config = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'max_epochs': 1000,
        'early_stopping_patience': 100,
        'hidden_dim': 128,
        'num_layers': 3,
        'dropout': 0.2,
        'node_dim': 5,
        'num_bead_types': 100,
        'embedding_dim': 32,
        'edge_dim': 3
    }
    
    print("Encapsulation Prediction GNN Training")
    
    seed = 121
    set_seed(seed)
    
    compounds_df = pd.read_csv("training_data.csv")
    compounds_df = compounds_df.dropna(subset=['encapsulation_mean'])
    print(f"\nLoaded {len(compounds_df)} compounds")
    
    nbfix_map = parse_nbfix_table("NBFIX_table")
    print(f"Loaded {len(nbfix_map)} bead type parameters")
    
    builder = MolecularGraphBuilder(nbfix_map, data_dir="training_data")
    graphs = builder.build_dataset(compounds_df)
    print(f"Built {len(graphs)} molecular graphs")
    
    if len(graphs) == 0:
        print("Error: No graphs were built!")
        return
    
    config['num_bead_types'] = builder.num_bead_types
    
    train_idx, temp_idx = train_test_split(range(len(graphs)), test_size=0.1, random_state=seed, shuffle=True)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed, shuffle=True)
    
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]
    
    print(f"\nTrain: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    
    train_loader = DataLoader(train_graphs, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=config['batch_size'], shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print("\nTraining...")
    
    model = EncapsulationGNN(
        node_dim=config['node_dim'],
        edge_dim=config['edge_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        num_bead_types=config['num_bead_types'],
        embedding_dim=config['embedding_dim']
    ).to(device)
    
    val_metrics, val_preds, val_targets = train_model(model, train_loader, val_loader, device, config)
    
    test_metrics, test_preds, test_targets = validate(model, test_loader, device)
    
    print("RESULTS")
    print("\nValidation Set:")
    print(f"  MAE: {val_metrics['mae']:.4f}, RMSE: {np.sqrt(val_metrics['mse']):.4f}")
    print(f"  R²: {val_metrics['r2']:.4f}, Spearman: {val_metrics['spearman']:.4f}")
    
    print("\nTest Set:")
    print(f"  MAE: {test_metrics['mae']:.4f}, RMSE: {np.sqrt(test_metrics['mse']):.4f}")
    print(f"  R²: {test_metrics['r2']:.4f}, Spearman: {test_metrics['spearman']:.4f}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    
    with open(f"results/results_{timestamp}.json", 'w') as f:
        json.dump({'config': config, 'val_metrics': val_metrics, 'test_metrics': test_metrics}, f, indent=2)
    
    pd.DataFrame({'target': val_targets, 'predicted': val_preds}).to_csv(
        f"results/val_predictions_{timestamp}.csv", index=False)
    pd.DataFrame({'target': test_targets, 'predicted': test_preds}).to_csv(
        f"results/test_predictions_{timestamp}.csv", index=False)
    
    torch.save(model.state_dict(), f"results/model_{timestamp}.pth")
    
    print(f"\nResults saved to results/results_{timestamp}.json")
    print(f"Model saved to results/model_{timestamp}.pth")


if __name__ == "__main__":
    main()
