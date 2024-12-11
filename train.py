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
import pickle
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append(os.path.abspath('./dataprocessing'))

from utils import load_path_data, normalize_features, set_deterministic, construct_small_graph, run_evaluate, construct_graph
from model import GTTP

def log_metrics(epoch, train_loss, val_loss, val_acc):
    """
    Logs training and validation losses to TensorBoard.

    Parameters:
    - epoch (int): Current epoch number
    - train_loss (float): The training loss for the current epoch
    - val_loss (float): The validation loss for the current epoch
    - val_acc (float): The validation accuracy for the current epoch
    """
    # Log the losses for train and validation
    # writer.add_scalar('Loss/train', train_loss, epoch)
    # writer.add_scalar('Loss/validation', val_loss, epoch)
    writer.add_scalars('Loss', {'train': train_loss, 'validation': val_loss}, epoch)
    writer.add_scalar('Accuracy/validation', val_acc, epoch)


if __name__ == "__main__":

    set_deterministic(100)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("trained_models", exist_ok=True)
    os.makedirs("graph_cache", exist_ok=True)

    num_anchor_points = 10

    if num_anchor_points > 0:
        mode = f"anchor_points_feature_{num_anchor_points}"
    else:
        mode = "gnn_embedding"

    model_file = f"trained_models/{mode}.pt"

    graph_path = "data/stanford.pbf"
    graph_fname = graph_path.split("/")[-1].split(".")[0]

    if os.path.exists(f"graph_cache/{graph_fname}_{num_anchor_points}.pkl"):
        with open(f"graph_cache/{graph_fname}_{num_anchor_points}.pkl", "rb") as f:
            graph, node_id_to_idx = pickle.load(f)
    else:
        graph, node_id_to_idx = construct_graph(graph_path, num_anchor_points=num_anchor_points)
        with open(f"graph_cache/{graph_fname}_{num_anchor_points}.pkl", "wb") as f:
            pickle.dump((graph, node_id_to_idx), f)

    graph.x = normalize_features(graph.x)
    graph.edge_attr = normalize_features(graph.edge_attr)

    graph.x = graph.x.to(device)
    graph.edge_index = graph.edge_index.to(device)
    graph.edge_attr = graph.edge_attr.to(device)

    # Load data and graph
    batch_size = 64
    route_dir = "dataprocessing/out"
    train_dataset, val_dataset, test_dataset = load_path_data(route_dir=route_dir, node_id_to_idx=node_id_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"train_dataset length: {len(train_dataset)}")
    print(f"val_dataset length: {len(val_dataset)}")
    writer = SummaryWriter(log_dir=f'runs/{mode}')

    embed_dim = 128
    gtn_hidden_dim = 32

    if mode == "gnn_embedding":
        input_dim = 2
    elif "anchor_points_feature" in mode:
        input_dim = 2 + num_anchor_points

    model = GTTP(input_dim, gtn_hidden_dim, embed_dim, gtn_layers=3, dropout=0.1)
    model = model.to(device)

    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    # Define the number of epochs
    num_epochs = 1000

    _, highest_val_acc = run_evaluate(model, graph, val_loader, device="cuda")

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
                    loss.backward()

                    acc_loss += loss * start_idx.size(0)
                    num_samples += start_idx.size(0)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Accumulate loss for tracking
            total_loss += acc_loss.item()

        # Print epoch loss
        avg_loss = total_loss / num_samples
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        val_loss, val_acc = run_evaluate(model, graph, val_loader, device="cuda")

        if val_acc > highest_val_acc:
            highest_val_acc = val_acc
            torch.save(model.state_dict(), model_file)

        log_metrics(epoch, train_loss=avg_loss, val_loss=val_loss, val_acc=val_acc)

    # Close the writer
    writer.close()