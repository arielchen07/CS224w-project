import os
import json
import math
import torch
import random
import osmium
from osmium import osm
from typing import List, Tuple
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.loader import NeighborLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import random
import numpy as np

from utils import load_path_data, normalize_features, set_deterministic, construct_small_graph, run_evaluate, construct_graph
from model import GTTP

if __name__ == "__main__":

    set_deterministic(100)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    modes = ["mlp_only", "gnn_embedding", "anchor_points_feature"]
    mode = "mlp_only"

    graph_path = "data/ev.pbf"
    graph, node_id_to_idx = construct_graph(graph_path)

    graph.x = normalize_features(graph.x)
    graph.edge_attr = normalize_features(graph.edge_attr)

    graph.x = graph.x.to(device)
    graph.edge_index = graph.edge_index.to(device)
    graph.edge_attr = graph.edge_attr.to(device)

    # Load data and graph
    batch_size = 64
    route_dir = "dataprocessing/outEv"
    train_dataset, val_dataset, test_dataset = load_path_data(route_dir=route_dir, node_id_to_idx=node_id_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"train_dataset length: {len(train_dataset)}")
    print(f"val_dataset length: {len(val_dataset)}")

    embed_dim = 256
    gtn_hidden_dim = 256
    disable_gnn = mode == "mlp_only"

    if mode in ["mlp_only", "gnn_embedding"]:
        input_dim = 2
    else:
        input_dim = 2 + 100

    model = GTTP(input_dim, gtn_hidden_dim, embed_dim, gtn_layers=5, dropout=0.1, disable_gnn=disable_gnn)
    model = model.to(device)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Define the number of epochs
    num_epochs = 10000

    run_evaluate(model, graph, val_loader, device="cuda")

    # Training loop
    for epoch in range(num_epochs):

        model.train()

        total_loss = 0
        num_samples = 0

        for batch in train_loader:

            start_idx, waypoints_shuffled, waypoints_correct, end_idx = [x.to(device) for x in batch]

            num_waypoints = waypoints_shuffled.size(1)

            acc_loss = 0

            # Backpropagation
            optimizer.zero_grad()

            for i in range(num_waypoints):
                for j in range(num_waypoints):

                    if i == j:
                        continue

                    waypoints_correct_cpy = waypoints_correct.clone().detach()

                    labels = waypoints_correct_cpy[:, i] < waypoints_correct_cpy[:, j]
                    preds = model(graph.x, graph.edge_index, graph.edge_attr, start_idx, end_idx, waypoints_shuffled[:, i], waypoints_shuffled[:, j])

                    loss = F.binary_cross_entropy_with_logits(preds.squeeze(1), labels.float())

                    acc_loss += loss * start_idx.size(0)
                    num_samples += start_idx.size(0)

            loss.backward()

            # # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Accumulate loss for tracking
            total_loss += acc_loss.item()

        # Print epoch loss
        avg_loss = total_loss / num_samples
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        run_evaluate(model, graph, val_loader, device="cuda")