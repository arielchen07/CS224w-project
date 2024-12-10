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

from utils import construct_graph, normalize_features

# Set the random seed for reproducibility
seed = 42  # Replace with your desired seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Enable deterministic mode
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class GTN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, beta=True, heads=1):
        super(GTN, self).__init__()

        self.num_layers = num_layers

        # Initialize transformer convolution layers with edge attributes
        conv_layers = [TransformerConv(input_dim, hidden_dim // heads, heads=heads, edge_dim=1, beta=beta)]
        conv_layers += [TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=1, beta=beta) for _ in range(num_layers - 2)]
        conv_layers.append(TransformerConv(hidden_dim, hidden_dim, heads=heads, edge_dim=1, beta=beta, concat=True))
        self.convs = torch.nn.ModuleList(conv_layers)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * heads, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, output_dim - input_dim)
        )

        # Initialize LayerNorm layers for normalization
        norm_layers = [torch.nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)]
        self.norms = torch.nn.ModuleList(norm_layers)

        self.dropout = dropout
        self.reset_parameters()

        self.adj_decoder = torch.nn.Linear(output_dim * 2, 1)
        self.edge_attr_decoder = torch.nn.Linear(output_dim * 2, 1)

        # self.adj_decoder = torch.nn.Sequential(
        #     torch.nn.Linear(output_dim * 2 * heads, hidden_dim),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(hidden_dim, 1)
        # )
        # self.edge_attr_decoder = torch.nn.Sequential(
        #     torch.nn.Linear(output_dim * 2 * heads, hidden_dim),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(hidden_dim, 1)
        # )

    def reset_parameters(self):
        """Resets parameters for the convolutional and normalization layers."""
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass with edge attributes.
        - x: Node features
        - edge_index: Edge indices
        - edge_attr: Edge attributes
        """
        # for i in range(self.num_layers - 1):
        #     x = self.convs[i](x, edge_index, edge_attr)  # Include edge_attr
        #     x = self.norms[i](x)
        #     x = F.relu(x)
        #     x = F.dropout(x, p=self.dropout, training=self.training)

        # # Last layer, average multi-head output.
        # x = self.convs[-1](x, edge_index, edge_attr)  # Include edge_attr

        # return x

        input_x = x

        for i in range(self.num_layers - 1):

            x = self.convs[i](x, edge_index, edge_attr)  # Graph convolution
            x = self.norms[i](x)  # Layer normalization
            x = F.relu(x)  # Non-linear activation
            x = F.dropout(x, p=self.dropout)  # Dropout

        # Last layer, no residual connection
        x = self.convs[-1](x, edge_index, edge_attr)
        
        x = self.mlp(x)
        x = torch.cat([input_x, x], dim=-1)

        return x

    
    def reconstruct_edge_attrs(self, node_embeddings, edge_index):
        """
        Reconstruct edge attributes using the node embeddings.
        - node_embeddings: Node embeddings learned from the forward pass.
        - edge_index: Edge indices for which attributes are reconstructed.
        """
        src_nodes = node_embeddings[edge_index[0]]  # Source node embeddings
        dst_nodes = node_embeddings[edge_index[1]]  # Destination node embeddings
        edge_features = torch.cat([src_nodes, dst_nodes], dim=-1)  # Concatenate node embeddings
        reconstructed_edge_attrs = self.edge_attr_decoder(edge_features)  # Predict edge attributes
        return reconstructed_edge_attrs

    def reconstruct_adj_matrix(self, node_embeddings):
        """
        Reconstruct the adjacency matrix using the node embeddings.
        - node_embeddings: Node embeddings learned from the forward pass.
        """
        num_nodes = node_embeddings.size(0)
        # Generate all possible pairs of nodes
        src_indices, dst_indices = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing='ij')
        src_nodes = node_embeddings[src_indices.flatten()]
        dst_nodes = node_embeddings[dst_indices.flatten()]
        edge_features = torch.cat([src_nodes, dst_nodes], dim=-1)
        adj_scores = self.adj_decoder(edge_features).view(num_nodes, num_nodes)  # Reshape to adjacency matrix size
        return torch.sigmoid(adj_scores)

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    model = GTN(input_dim=2, hidden_dim=128, output_dim=128, num_layers=3, dropout=0.1, beta=True, heads=1)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    epochs = 1000

    min_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_adj_loss = 0
        epoch_edge_attr_loss = 0

        optimizer.zero_grad()
        node_embeddings = model(graph.x, graph.edge_index, graph.edge_attr)

        # Reconstruct edge attributes
        predicted_edge_attrs = model.reconstruct_edge_attrs(node_embeddings, graph.edge_index)

        # Reconstruct adjacency matrix
        predicted_adj = model.reconstruct_adj_matrix(node_embeddings)

        if len(predicted_edge_attrs) > 0:
            edge_attr_loss = F.mse_loss(predicted_edge_attrs, graph.edge_attr)
        else:
            edge_attr_loss = 0

        adj_loss = F.binary_cross_entropy(predicted_adj, adj_matrix)

        loss = edge_attr_loss + adj_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        epoch_adj_loss += adj_loss.item()
        epoch_edge_attr_loss += edge_attr_loss.item()

        # Print training progress
        print(f"Epoch {epoch+1}/{epochs} "
                f"Edge Attr Loss: {epoch_edge_attr_loss:.4f} Adj Loss: {epoch_adj_loss:.4f}")
        
        if epoch_edge_attr_loss + epoch_adj_loss < min_loss:
            min_loss = epoch_edge_attr_loss + epoch_adj_loss
            torch.save(model.state_dict(), 'gtn.pth')
            print("Saving model to gtn.pth")