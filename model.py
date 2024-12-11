import torch
from torch_geometric.nn import TransformerConv
import torch.nn.functional as F
import torch.nn as nn
import torch


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
            torch.nn.Linear(hidden_dim, output_dim)
        )

        self.input_proj = torch.nn.Linear(input_dim, output_dim)

        # Initialize LayerNorm layers for normalization
        norm_layers = [torch.nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)]
        self.norms = torch.nn.ModuleList(norm_layers)

        self.dropout = dropout
        self.reset_parameters()

        self.adj_decoder = torch.nn.Linear(output_dim * 2, 1)
        self.edge_attr_decoder = torch.nn.Linear(output_dim * 2, 1)


    def reset_parameters(self):
        """Resets parameters for the convolutional and normalization layers."""
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        torch.nn.init.xavier_uniform_(self.input_proj.weight)
        torch.nn.init.zeros_(self.input_proj.bias)


    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass with edge attributes.
        - x: Node features
        - edge_index: Edge indices
        - edge_attr: Edge attributes
        """

        input_x = x
        
        for i in range(self.num_layers - 1):

            x = self.convs[i](x, edge_index, edge_attr)  # Graph convolution
            x = F.relu(x)  # Non-linear activation
            x = self.norms[i](x)  # Layer normalization
            x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout

        # Last layer, no residual connection
        x = self.convs[-1](x, edge_index, edge_attr)
        
        x = self.mlp(x)
        x = x + self.input_proj(input_x)

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


class NodeTransformer(nn.Module):
    def __init__(self, embed_dim, hidden_dim=128, dropout=0.1):
        super(NodeTransformer, self).__init__()
        
        # LayerNorm to stabilize the output
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.head = nn.Linear(hidden_dim, 1)


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
    def __init__(self, input_dim, gtn_hidden_dim, embed_dim, gtn_layers=3, dropout=0.1, disable_gnn=False):
        super(GTTP, self).__init__()
        self.disable_gnn = disable_gnn

        if self.disable_gnn:
            self.node_transformer_model = NodeTransformer(embed_dim=input_dim)
        else:
            self.gtn = GTN(input_dim=input_dim, hidden_dim=gtn_hidden_dim, output_dim=embed_dim, num_layers=gtn_layers, dropout=dropout, beta=True, heads=1)
            self.node_transformer_model = NodeTransformer(embed_dim=embed_dim)


    def forward(self, x, edge_index, edge_attr, start_idx, end_idx, x_1, x_2):

        if self.disable_gnn:
            node_embeddings = x
        else:
            node_embeddings = self.gtn(x, edge_index, edge_attr)
        start_node_embed = node_embeddings[start_idx].unsqueeze(1)
        end_node_embed = node_embeddings[end_idx].unsqueeze(1)
        x_1_embedding = node_embeddings[x_1]
        x_2_embedding = node_embeddings[x_2]

        pred = self.node_transformer_model(x_1_embedding, x_2_embedding, start_node_embed, end_node_embed)

        return pred
