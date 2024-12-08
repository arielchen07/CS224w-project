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
        conv_layers.append(TransformerConv(hidden_dim, output_dim, heads=heads, edge_dim=1, beta=beta, concat=True))
        self.convs = torch.nn.ModuleList(conv_layers)

        # Initialize LayerNorm layers for normalization
        norm_layers = [torch.nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)]
        self.norms = torch.nn.ModuleList(norm_layers)

        self.dropout = dropout
        self.reset_parameters()

        # self.adj_decoder = torch.nn.Linear(output_dim * 2 * heads, 1)
        # self.edge_attr_decoder = torch.nn.Linear(output_dim * 2 * heads, 1)

        self.adj_decoder = torch.nn.Sequential(
            torch.nn.Linear(output_dim * 2 * heads, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
        self.edge_attr_decoder = torch.nn.Sequential(
            torch.nn.Linear(output_dim * 2 * heads, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

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

        for i in range(self.num_layers - 1):

            # if i > 0:
            #     residual = x  # Save the input for the residual connection
            
            x = self.convs[i](x, edge_index, edge_attr)  # Graph convolution
            x = self.norms[i](x)  # Layer normalization
            x = F.relu(x)  # Non-linear activation
            x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout

            # if i > 0:
            #     x = x + residual  # Add the residual connection

        # Last layer, no residual connection
        x = self.convs[-1](x, edge_index, edge_attr)

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

    graph_path = "data/stanford.pbf"
    graph, node_id_to_idx = construct_graph(graph_path)
    graph.x = graph.x.to(device)
    graph.edge_index = graph.edge_index.to(device)
    graph.edge_attr = graph.edge_attr.to(device)

    graph.x = normalize_features(graph.x)
    graph.edge_attr = normalize_features(graph.edge_attr)

    adj_matrix = torch.zeros((len(graph.x), len(graph.x)), dtype=torch.float32).to(device)
    adj_matrix[graph.edge_index[0], graph.edge_index[1]] = 1.0

    batch_size = 64  # Adjust based on available memory
    dataloader = NeighborLoader(
        data=graph,  # Your large graph
        num_neighbors=[10, 10],  # Number of neighbors to sample at each layer
        batch_size=batch_size,  # Number of root nodes per batch
        shuffle=True,  # Shuffle the data
    )

    model = GTN(input_dim=2, hidden_dim=1024, output_dim=1024, num_layers=10, dropout=0.1, beta=True, heads=4)
    model = model.to(device)

    # def initialize_weights(module):
    #     if isinstance(module, torch.nn.Linear):
    #         torch.nn.init.xavier_uniform_(module.weight)
    #         if module.bias is not None:
    #             torch.nn.init.zeros_(module.bias)

    # model.apply(initialize_weights)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)

    epochs = 0

    # for idx, batch in enumerate(dataloader):
    #     batch = batch.to(device)
    #     break

    min_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_adj_loss = 0
        epoch_edge_attr_loss = 0

        for idx, batch in enumerate(dataloader):

            batch = batch.to(device)

        # for i in range(1):

            optimizer.zero_grad()
            node_embeddings = model(batch.x, batch.edge_index, batch.edge_attr)

            # Reconstruct edge attributes
            predicted_edge_attrs = model.reconstruct_edge_attrs(node_embeddings, batch.edge_index)

            # Reconstruct adjacency matrix
            predicted_adj = model.reconstruct_adj_matrix(node_embeddings)

            # Generate ground truth adjacency matrix for the batch
            batch_adj_matrix = torch.zeros((len(batch.x), len(batch.x)), dtype=torch.float32).to(device)
            batch_adj_matrix[batch.edge_index[0], batch.edge_index[1]] = 1.0

            if len(predicted_edge_attrs) > 0:
                edge_attr_loss = F.mse_loss(predicted_edge_attrs, batch.edge_attr)
            else:
                edge_attr_loss = 0

            adj_loss = F.binary_cross_entropy(predicted_adj, batch_adj_matrix)

            loss = edge_attr_loss + adj_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # scheduler.step(loss)

            # if epoch == epochs - 1 and idx == 0:

            #     torch.set_printoptions(threshold=torch.inf)

            #     print(f"predicted_edge_attrs: {predicted_edge_attrs}")
            #     print(f"batch.edge_attr: {batch.edge_attr}")

            #     print(f"predicted_adj: {predicted_adj}")
            #     print(f"batch_adj_matrix: {batch_adj_matrix}")

            #     torch.set_printoptions()

            epoch_adj_loss += edge_attr_loss.item()
            epoch_edge_attr_loss += adj_loss.item()

        # Print training progress
        print(f"Epoch {epoch+1}/{epochs} "
                f"Edge Attr Loss: {epoch_edge_attr_loss / len(batch):.4f} Adj Loss: {epoch_adj_loss / len(batch):.4f}")
        
        if epoch_edge_attr_loss + epoch_adj_loss < min_loss:
            min_loss = epoch_edge_attr_loss + epoch_adj_loss
            torch.save(model.state_dict(), 'gtn.pth')

    model = GTN(input_dim=2, hidden_dim=1024, output_dim=1024, num_layers=10, dropout=0.1, beta=True, heads=4)
    model = model.to(device)

    # Load the state dictionary into the model
    model.load_state_dict(torch.load('gtn.pth'))

    # Switch the model to evaluation mode (optional, for inference)
    model.eval()

    epoch_adj_loss = 0
    epoch_edge_attr_loss = 0

    for idx, batch in enumerate(dataloader):

        batch = batch.to(device)

        node_embeddings = model(batch.x, batch.edge_index, batch.edge_attr)

        # Reconstruct edge attributes
        predicted_edge_attrs = model.reconstruct_edge_attrs(node_embeddings, batch.edge_index)

        # Reconstruct adjacency matrix
        predicted_adj = model.reconstruct_adj_matrix(node_embeddings)

        # Generate ground truth adjacency matrix for the batch
        batch_adj_matrix = torch.zeros((len(batch.x), len(batch.x)), dtype=torch.float32).to(device)
        batch_adj_matrix[batch.edge_index[0], batch.edge_index[1]] = 1.0

        if len(predicted_edge_attrs) > 0:
            edge_attr_loss = F.mse_loss(predicted_edge_attrs, batch.edge_attr)
        else:
            edge_attr_loss = 0

        adj_loss = F.binary_cross_entropy(predicted_adj, batch_adj_matrix)

        loss = edge_attr_loss + adj_loss

        epoch_adj_loss += edge_attr_loss.item()
        epoch_edge_attr_loss += adj_loss.item()

    print(node_embeddings)

    # Print training progress
    print(f"Evaluation: Edge Attr Loss: {epoch_edge_attr_loss / len(batch):.4f} Adj Loss: {epoch_adj_loss / len(batch):.4f}")