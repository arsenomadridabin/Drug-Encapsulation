import os
import sys
import argparse
import json
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from parse_itp import parse_nbfix_table
from build_graphs import MolecularGraphBuilder
from gnn_model import EncapsulationGNN


def load_model(model_path, config_path, device):
    with open(config_path, 'r') as f:
        config = json.load(f)['config']
    
    model = EncapsulationGNN(
        node_dim=config['node_dim'],
        edge_dim=config['edge_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        num_bead_types=config['num_bead_types'],
        embedding_dim=config['embedding_dim']
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict(model, graphs, device):
    loader = DataLoader(graphs, batch_size=32, shuffle=False)
    preds, ids = [], []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                       batch.bead_type_id, batch.num_atoms, batch.num_bonds,
                       batch.total_mass, batch.avg_degree, batch.max_degree, batch.min_degree,
                       batch.degree_std, batch.num_leaves, batch.graph_density,
                       batch.avg_bond_length, batch.bond_length_std, batch.avg_force_constant,
                       batch.total_charge, batch.charge_std, batch.unique_bead_types,
                       batch.bead_type_diversity, batch.mass_std)
            preds.extend(out.cpu().numpy().flatten().tolist())
            ids.extend(batch.compound_id)
    
    return ids, preds


def find_compounds_in_folder(folder_path):
    compounds = []
    if not os.path.exists(folder_path):
        return compounds
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path) and any(f.endswith('.itp') for f in os.listdir(item_path)):
            compounds.append(item)
    return sorted(compounds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compounds", "-c", nargs="+", help="Compound IDs")
    parser.add_argument("--file", "-f", help="CSV file with compound column")
    parser.add_argument("--folder", "-d", help="Folder with compound subdirectories")
    parser.add_argument("--model", "-m", required=True, help="Model checkpoint path")
    parser.add_argument("--config", required=True, help="Config JSON path")
    parser.add_argument("--nbfix", default="NBFIX_table", help="NBFIX table file path")
    parser.add_argument("--output", "-o", default="predictions.csv", help="Output CSV")
    
    args = parser.parse_args()
    
    if args.folder:
        compound_ids = find_compounds_in_folder(args.folder)
        data_dir = args.folder
    elif args.compounds:
        compound_ids = args.compounds
        data_dir = "training_data"
    elif args.file:
        df = pd.read_csv(args.file)
        if 'compound' not in df.columns:
            print("Error: CSV must have 'compound' column")
            sys.exit(1)
        compound_ids = df['compound'].tolist()
        data_dir = "training_data"
    else:
        print("Error: Provide --folder, --compounds, or --file")
        parser.print_help()
        sys.exit(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model, args.config, device)
    
    nbfix_map = parse_nbfix_table(args.nbfix)
    builder = MolecularGraphBuilder(nbfix_map, data_dir=data_dir)
    
    graphs = []
    for compound_id in compound_ids:
        graph = builder.build_graph(compound_id)
        if graph:
            graphs.append(graph)
    
    if not graphs:
        print("Error: No graphs built")
        sys.exit(1)
    
    compound_ids_pred, predictions = predict(model, graphs, device)
    
    pd.DataFrame({
        'compound': compound_ids_pred,
        'predicted_encapsulation': predictions
    }).to_csv(args.output, index=False)
    
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
