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
import networkx as nx
import torch.nn as nn
import torch.optim as optim
from math import radians, cos
import torch
import random
import numpy as np
# from torch_sparse import SparseTensor

from utils import construct_graph, load_path_data, normalize_features
from train_gtn import GTN

# Set the random seed for reproducibility
seed = 42  # Replace with your desired seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Enable deterministic mode
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"

# graph_path = "data/stanford.pbf"
# graph, node_id_to_idx = construct_graph(graph_path)

nodes = []
for i in range(10):
    for j in range(10):
        nodes.append([i, j])

x = torch.tensor(nodes, dtype=torch.float)

edge_index_list = []
edge_attr_list = []
for i in range(10):
    for j in range(10):
        if i < 9:
            edge_index_list.append([i*10+j, (i+1)*10+j])
            edge_attr_list.append(1.0)
        if j < 9:
            edge_index_list.append([i*10+j, i*10+j+1])
            edge_attr_list.append(1.0)

edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_attr_list, dtype=torch.float).unsqueeze(1)

graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

graph.x = graph.x.to(device)
graph.edge_index = graph.edge_index.to(device)
graph.edge_attr = graph.edge_attr.to(device)

graph.x = normalize_features(graph.x)
graph.edge_attr = normalize_features(graph.edge_attr)

adj_matrix = torch.zeros((len(graph.x), len(graph.x)), dtype=torch.float32).to(device)
adj_matrix[graph.edge_index[0], graph.edge_index[1]] = 1.0

print(f"graph.x.shape: {graph.x.shape}")
print(f"graph.edge_index.shape: {graph.edge_index.shape}")
print(f"graph.edge_attr.shape: {graph.edge_attr.shape}")

class NodeTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, max_seq_len=48, ff_dim=64, dropout=0.1):
        super(NodeTransformer, self).__init__()
        
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        
        # Learnable special embeddings for fixed start and end node
        self.start_node_embed_tag = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=False)
        self.end_node_embed_tag = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=False)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=num_heads, 
                dim_feedforward=ff_dim, 
                dropout=dropout, 
                activation='gelu',
                batch_first=True,
            ) 
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        
        # LayerNorm to stabilize the output
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        self.head = nn.Linear(embed_dim, 1)

    def forward(self, waypoint_node_embeds, start_node_embed, end_node_embed):
        """
        Args:
            waypoint_node_embeds: Tensor of shape (batch_size, seq_len, embed_dim), where seq_len <= max_seq_len.
            start_node_embed: Tensor of shape (batch_size, 1, embed_dim) representing the first fixed node embedding.
            end_node_embed: Tensor of shape (batch_size, 1, embed_dim) representing the second fixed node embedding.

        Returns:
            Tensor of shape (batch_size, seq_len + 2, embed_dim).
        """

        batch_size, seq_len, embed_dim = waypoint_node_embeds.shape
        
        assert seq_len <= self.max_seq_len, f"Sequence length should be <= {self.max_seq_len}"
        assert embed_dim == self.embed_dim, f"Embedding dimension mismatch: {embed_dim} != {self.embed_dim}"

        # Add learnable tags to fixed nodes
        # start_node_embed = start_node_embed + self.start_node_embed_tag  # Shape: (batch_size, 1, embed_dim)
        # end_node_embed = end_node_embed + self.end_node_embed_tag  # Shape: (batch_size, 1, embed_dim)

        # Concatenate fixed nodes with the variable-length sequence
        fixed_nodes = torch.cat([start_node_embed, end_node_embed], dim=1)  # Shape: (batch_size, 2, embed_dim)
        full_sequence = torch.cat([fixed_nodes, waypoint_node_embeds], dim=1)  # Shape: (batch_size, seq_len+2, embed_dim)

        # Pass through the Transformer encoder layers
        x = full_sequence

        # print(f"waypoint_node_embeds: {waypoint_node_embeds}")

        # print(f"embedding: {x}")

        for layer in self.encoder_layers:
            x = layer(x)
        
        # Apply LayerNorm
        x = self.norm(x)
        x = self.mlp(x)
        x = self.head(x)
        
        return x

class GTTP(nn.Module):
    def __init__(self):
        super(GTTP, self).__init__()

        embed_dim = 128
        ff_dim = embed_dim
        hidden_dim = embed_dim

        self.gtn = GTN(input_dim=2, hidden_dim=hidden_dim, output_dim=embed_dim, num_layers=3, dropout=0.1, beta=True, heads=1)
        self.node_transformer_model = NodeTransformer(embed_dim=embed_dim, num_heads=1, num_layers=3, ff_dim=ff_dim, dropout=0.1)

    def forward(self, x, edge_index, edge_attr, start_idx, end_idx, waypoint_node_indices):

        node_embeddings = self.gtn(x, edge_index, edge_attr)
        start_node_embed = node_embeddings[start_idx].unsqueeze(1)
        end_node_embed = node_embeddings[end_idx].unsqueeze(1)
        waypoint_node_embeds = node_embeddings[waypoint_node_indices]
        pred = self.node_transformer_model(waypoint_node_embeds, start_node_embed, end_node_embed)

        return pred

def pairwise_ranking_loss(predicted_ordering, correct_ordering, margin=3.0):
    """
    Pairwise ranking loss for predicting the relative order of waypoints.

    Args:
        predicted_ordering (torch.Tensor): Predicted scores for each waypoint, shape (batch_size, seq_len).
        correct_ordering (torch.Tensor): Ground truth relative order, shape (batch_size, seq_len).
        margin (float): Margin for ranking loss.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    batch_size, seq_len = predicted_ordering.size()
    
    # Expand dimensions for pairwise comparison
    pred_diff = predicted_ordering.unsqueeze(2) - predicted_ordering.unsqueeze(1)  # Shape: (batch_size, seq_len, seq_len)
    true_diff = correct_ordering.unsqueeze(2) - correct_ordering.unsqueeze(1)      # Shape: (batch_size, seq_len, seq_len)
    
    # Compute pairwise labels (+1 if correct ordering, -1 otherwise)
    pairwise_labels = (true_diff > 0).float() * 2 - 1  # Shape: (batch_size, seq_len, seq_len)

    # Compute ranking loss for each pair
    loss = F.relu(margin - pred_diff * pairwise_labels)  # Shape: (batch_size, seq_len, seq_len)

    # Mask out diagonal entries (self-comparisons)
    mask = torch.eye(seq_len, dtype=torch.bool, device=predicted_ordering.device)
    loss = loss.masked_fill(mask.unsqueeze(0), 0.0)

    # Return mean loss
    return loss.sum() / (batch_size * (seq_len * (seq_len - 1)))

def run_evaluate(model, loader):
    model.eval()

    total_loss = 0
    num_batches = 0

    for batch in loader:
        start_idx, waypoints_shuffled, waypoints_correct, end_idx = [x.to(device) for x in batch]

        with torch.no_grad():
            predicted_ordering = model(
                graph.x, graph.edge_index, graph.edge_attr, start_idx, end_idx, waypoints_shuffled
            )

            predicted_ordering = predicted_ordering

            # Process the predictions
            predicted_ordering = predicted_ordering[:, 2:].squeeze(2)

            # Compute loss
            loss = loss_fn(predicted_ordering, waypoints_correct.float())

            # Accumulate loss for tracking
            total_loss += loss.item()
            num_batches += 1

    # Print epoch loss
    avg_loss = total_loss / num_batches
    print(f"Validation Loss: {avg_loss:.4f}")

    return avg_loss


node_id_to_idx = {}

for i in range(10):
    for j in range(10):
        node_id_to_idx[f"{i},{j}"] = i*10+j

# Load data and graph
batch_size = 64
route_dir = "dataprocessing/outSmall"
train_dataset, val_dataset, test_dataset = load_path_data(route_dir=route_dir, node_id_to_idx=node_id_to_idx)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"train_dataset length: {len(train_dataset)}")
print(f"val_dataset length: {len(val_dataset)}")
print(f"test_dataset length: {len(test_dataset)}")

model = GTTP()
model = model.to(device)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Define the number of epochs
num_epochs = 10000

# Define a proper loss function
loss_fn = pairwise_ranking_loss
# loss_fn = nn.MSELoss()

min_val_loss = run_evaluate(model, val_loader)

# Training loop
for epoch in range(num_epochs):

    model.train()

    total_loss = 0
    num_batches = 0

    for batch in train_loader:
    # for i in range(1):

        start_idx, waypoints_shuffled, waypoints_correct, end_idx = [x.to(device) for x in batch]

        # Forward pass
        predicted_ordering = model(
            graph.x, graph.edge_index, graph.edge_attr, start_idx, end_idx, waypoints_shuffled
        )

        predicted_ordering = predicted_ordering * 8

        # Process the predictions
        predicted_ordering = predicted_ordering[:, 2:].squeeze(2)

        # Compute loss
        loss = loss_fn(predicted_ordering, waypoints_correct.float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate loss for tracking
        total_loss += loss.item()
        num_batches += 1

    print(f"predicted_ordering: {predicted_ordering[0]}")
    print(f"waypoints_correct: {waypoints_correct[0]}")

    # Print epoch loss
    avg_loss = total_loss / num_batches
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    val_loss = run_evaluate(model, val_loader)

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(), 'gttp.pth')
        print(f"Model saved with loss: {min_val_loss}")