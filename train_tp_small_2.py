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
from train_gtn_small import GTN

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

        for ii in range(10):
            for jj in range(10):
                nodes[-1].append(abs(ii-i) + abs(jj-j))

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

class NodeTransformer(nn.Module):
    def __init__(self, embed_dim):
        super(NodeTransformer, self).__init__()
        
        # LayerNorm to stabilize the output
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
        )

        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x_1, x_2, start_node_embed, end_node_embed):

        start_node_embed = start_node_embed.squeeze(1)
        end_node_embed = end_node_embed.squeeze(1)

        # Concatenate fixed nodes with the variable-length sequence
        nodes = torch.cat([start_node_embed, end_node_embed, x_1, x_2], dim=1)  # Shape: (batch_size, 2, embed_dim)

        x = nodes
        x = self.mlp(x)
        x = self.head(x)
        
        return x

class GTTP(nn.Module):
    def __init__(self):
        super(GTTP, self).__init__()

        embed_dim = 512
        ff_dim = embed_dim
        hidden_dim = embed_dim

        self.gtn = GTN(input_dim=2+100, hidden_dim=embed_dim, output_dim=embed_dim, num_layers=3, dropout=0.1, beta=True, heads=1)
        self.node_transformer_model = NodeTransformer(embed_dim=embed_dim)

    def forward(self, x, edge_index, edge_attr, start_idx, end_idx, x_1, x_2):

        node_embeddings = self.gtn(x, edge_index, edge_attr)
        start_node_embed = node_embeddings[start_idx].unsqueeze(1)
        end_node_embed = node_embeddings[end_idx].unsqueeze(1)
        x_1_embedding = node_embeddings[x_1]
        x_2_embedding = node_embeddings[x_2]

        pred = self.node_transformer_model(x_1_embedding, x_2_embedding, start_node_embed, end_node_embed)

        return pred

def run_evaluate(model, loader):
    model.eval()

    total_loss = 0
    num_batches = 0
    total_num = 0
    correct_num = 0

    for batch in loader:

        start_idx, waypoints_shuffled, waypoints_correct, end_idx = [x.to(device) for x in batch]

        num_waypoints = waypoints_shuffled.size(1)

        for i in range(num_waypoints):
            for j in range(num_waypoints):

                if i == j:
                    continue

                with torch.no_grad():
                    
                    labels = waypoints_correct[:, i] < waypoints_correct[:, j]
                    
                    preds = model(graph.x, graph.edge_index, graph.edge_attr, start_idx, end_idx, waypoints_shuffled[:, i], waypoints_shuffled[:, j])

                    loss = F.binary_cross_entropy_with_logits(preds.squeeze(1), labels.float())

                    preds = torch.sigmoid(preds).squeeze(1)
                    preds = (preds > 0.5).float()
                    correct_num += (preds == labels).sum().item()
                    total_num += len(labels)

            # Accumulate loss for tracking
            total_loss += loss.item()
            num_batches += 1

    # Print epoch loss
    avg_loss = total_loss / num_batches
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Validation Accuracy: {correct_num / total_num:.4f}")

    return avg_loss


node_id_to_idx = {}

for i in range(10):
    for j in range(10):
        node_id_to_idx[f"{i},{j}"] = i*10+j

# Load data and graph
batch_size = 64
route_dir = "dataprocessing/outSmall"
train_dataset, val_dataset, test_dataset = load_path_data(route_dir=route_dir, node_id_to_idx=node_id_to_idx)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"train_dataset length: {len(train_dataset)}")
print(f"val_dataset length: {len(val_dataset)}")
print(f"test_dataset length: {len(test_dataset)}")

model = GTTP()

# model.gtn.load_state_dict(torch.load('gtn.pth'))

model = model.to(device)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Define the number of epochs
num_epochs = 10000

min_val_loss = run_evaluate(model, val_loader)

# Training loop
for epoch in range(num_epochs):

    model.train()

    total_loss = 0
    num_batches = 0

    for batch in train_loader:
    # for i in range(1):

        start_idx, waypoints_shuffled, waypoints_correct, end_idx = [x.to(device) for x in batch]

        num_waypoints = waypoints_shuffled.size(1)

        for i in range(num_waypoints):
            for j in range(i + 1, num_waypoints):

                waypoints_correct_cpy = waypoints_correct.clone().detach()

                labels = waypoints_correct_cpy[:, i] < waypoints_correct_cpy[:, j]
                preds = model(graph.x, graph.edge_index, graph.edge_attr, start_idx, end_idx, waypoints_shuffled[:, i], waypoints_shuffled[:, j])

                loss = F.binary_cross_entropy_with_logits(preds.squeeze(1), labels.float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate loss for tracking
        total_loss += loss.item()
        num_batches += 1

    # Print epoch loss
    avg_loss = total_loss / num_batches
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    val_loss = run_evaluate(model, val_loader)

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(), 'gttp.pth')
        print(f"Model saved with loss: {min_val_loss}")