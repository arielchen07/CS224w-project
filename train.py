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
import torch.nn.functional as F
import torch.nn as nn
import networkx as nx

import torch
import random
import numpy as np

# Set the random seed for reproducibility
seed = 42  # Replace with your desired seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Enable deterministic mode
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


R = 6371000  

from math import radians, cos

anchor_point = (37.4340414, -122.17246)

def xy_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the x-distance (longitude) and y-distance (latitude) in meters
    between two geographic points on Earth.
    """
    # Earth radius in meters
    R = 6371000  
    
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Calculate the y-distance (latitude difference in meters)
    delta_lat = lat2 - lat1
    y_distance = delta_lat * R
    
    # Calculate the x-distance (longitude difference in meters)
    delta_lon = lon2 - lon1
    x_distance = delta_lon * R * cos((lat1 + lat2) / 2)  # Adjust for latitude

    return x_distance, y_distance

def construct_graph(osmPath: str):

    def compute_distance(lat1, lon1, lat2, lon2):
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = (math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    class MapCreationHandler(osmium.SimpleHandler):
        def __init__(self) -> None:
            super().__init__()
            self.nodes = []
            self.edges = [[], []]
            self.edge_dist = []
            self.node_id_to_idx = {}
            self.idx_to_node_id = {}
            self.id_counter = 0

        def node(self, n: osmium.osm.Node) -> None:
            xy = xy_distance(n.location.lat, n.location.lon, anchor_point[0], anchor_point[1])
            self.nodes.append([xy[0], xy[1]])
            self.node_id_to_idx[n.id] = self.id_counter
            self.idx_to_node_id[self.id_counter] = n.id
            self.id_counter += 1

        def way(self, w):
            node_refs = [node.ref for node in w.nodes]

            for i in range(len(node_refs) - 1):
                node_start = node_refs[i]
                node_end = node_refs[i + 1]

                node_1_idx = self.node_id_to_idx[node_start]
                node_2_idx = self.node_id_to_idx[node_end]

                self.edges[0].append(node_1_idx)
                self.edges[1].append(node_2_idx)

                node_1 = self.nodes[node_1_idx]
                node_2 = self.nodes[node_2_idx]

                n1_lat, n1_lon = node_1
                n2_lat, n2_lon = node_2

                dist = compute_distance(n1_lat, n1_lon, n2_lat, n2_lon)
                self.edge_dist.append(dist)

    mapCreator = MapCreationHandler()
    mapCreator.apply_file(osmPath, locations=True)

    x = torch.tensor(mapCreator.nodes, dtype=torch.float)
    edge_index = torch.tensor(mapCreator.edges, dtype=torch.long)
    edge_attr = torch.tensor(mapCreator.edge_dist, dtype=torch.float).unsqueeze(1)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data, mapCreator.node_id_to_idx

class PathDataset(Dataset):

    def __init__(self, route_files: List[str], node_id_to_idx: dict, shuffle_waypoints: bool = True, fixed_length: int = 8):
        self.route_files = route_files
        self.node_id_to_idx = node_id_to_idx
        self.shuffle_waypoints = shuffle_waypoints
        self.fixed_length = fixed_length
        self.data = self._load_data()

    def _load_data(self):
        samples = []
        for f in self.route_files:
            with open(f, 'r') as json_file:
                route_info = json.load(json_file)

            start_id = route_info["start"]
            end_id = route_info["end"]
            waypoint_tags = route_info["waypointTags"]
            waypoint_ids = [tag.split('=')[1] for tag in waypoint_tags]

            try:
                start_idx = self.node_id_to_idx[int(start_id)]
                end_idx = self.node_id_to_idx[int(end_id)]
                waypoints_correct = [self.node_id_to_idx[int(w_id)] for w_id in waypoint_ids]
            except KeyError:
                continue

            samples.append({
                "start_idx": start_idx,
                "waypoints_correct": waypoints_correct,
                "end_idx": end_idx
            })
        return samples
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        start_idx = torch.tensor(sample["start_idx"], dtype=torch.long)
        end_idx = torch.tensor(sample["end_idx"], dtype=torch.long)
        waypoints_correct = sample["waypoints_correct"]

        # Pad way points if length is different
        if len(waypoints_correct) < self.fixed_length:
            padding_needed = self.fixed_length - len(waypoints_correct)
            waypoints_correct = waypoints_correct + [-1] * padding_needed

        waypoints_correct_tensor = torch.tensor(waypoints_correct, dtype=torch.long)

        # Generate indices for the original order
        original_order = torch.arange(len(waypoints_correct), dtype=torch.long)

        if self.shuffle_waypoints and len(waypoints_correct) > 1:
            waypoints_shuffled = waypoints_correct[:]
            
            # Randomly shuffle waypoints
            random.shuffle(waypoints_shuffled)

            # Create a mapping of shuffled indices
            shuffled_indices = [waypoints_correct.index(wp) for wp in waypoints_shuffled]

            waypoints_shuffled_tensor = torch.tensor(waypoints_shuffled, dtype=torch.long)
            shuffled_order_tensor = torch.tensor(shuffled_indices, dtype=torch.long)
        else:
            waypoints_shuffled_tensor = waypoints_correct_tensor.clone()
            shuffled_order_tensor = original_order.clone()

        return start_idx, waypoints_shuffled_tensor, shuffled_order_tensor, end_idx
    
def load_path_data(route_dir: str, node_id_to_idx: dict, train_ratio=0.8, val_ratio=0.1, seed=42):

    torch.manual_seed(seed)

    route_files = [os.path.join(route_dir, f) for f in os.listdir(route_dir) if f.endswith('.json')]
    full_dataset = PathDataset(route_files, node_id_to_idx, shuffle_waypoints=False)
    dataset_len = len(full_dataset)
    train_len = int(dataset_len * train_ratio)
    val_len = int(dataset_len * val_ratio)
    test_len = dataset_len - train_len - val_len

    train_dataset_raw, val_dataset_raw, test_dataset_raw = random_split(full_dataset, [train_len, val_len, test_len])

    train_route_files = [route_files[i] for i in train_dataset_raw.indices]
    val_route_files = [route_files[i] for i in val_dataset_raw.indices]
    test_route_files = [route_files[i] for i in test_dataset_raw.indices]

    train_dataset = PathDataset(train_route_files, node_id_to_idx, shuffle_waypoints=True)
    val_dataset = PathDataset(val_route_files, node_id_to_idx, shuffle_waypoints=False)
    test_dataset = PathDataset(test_route_files, node_id_to_idx, shuffle_waypoints=False)

    return train_dataset, val_dataset, test_dataset

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
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_attr)  # Include edge_attr
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Last layer, average multi-head output.
        x = self.convs[-1](x, edge_index, edge_attr)  # Include edge_attr

        return x
    
class NodeTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, max_seq_len=48, ff_dim=64, dropout=0.1):
        super(NodeTransformer, self).__init__()
        
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        
        # Learnable special embeddings for fixed start and end node
        self.start_node_embed_tag = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)
        self.end_node_embed_tag = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)
        
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
        
        # LayerNorm to stabilize the output
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
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

        for layer in self.encoder_layers:
            x = layer(x)
        
        # Apply LayerNorm
        x = self.norm(x)

        x = self.linear(x)
        x = F.relu(x)
        x = self.head(x)
        
        return x

# Load data and graph
batch_size = 1
graph_path = "data/stanford.pbf"
route_dir = "dataprocessing/out"
graph, node_id_to_idx = construct_graph(graph_path)
train_dataset, val_dataset, test_dataset = load_path_data(route_dir=route_dir, node_id_to_idx=node_id_to_idx)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"train_dataset length: {len(train_dataset)}")
print(f"val_dataset length: {len(val_dataset)}")
print(f"test_dataset length: {len(test_dataset)}")


class GTTP(nn.Module):
    def __init__(self):
        super(GTTP, self).__init__()

        embed_dim = 1024

        self.gtn = GTN(input_dim=2, hidden_dim=10, output_dim=embed_dim, num_layers=2, dropout=0, beta=True, heads=1)
        self.node_transformer_model = NodeTransformer(embed_dim=embed_dim, num_heads=1, num_layers=1, ff_dim=1024, dropout=0)

        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
        self.apply(init_weights)

        for param in self.gtn.parameters():
            param.requires_grad = False

    def forward(self, x, edge_index, edge_attr, start_idx, end_idx, waypoint_node_indices):

        node_embeddings = self.gtn(x, edge_index, edge_attr)
        start_node_embed = node_embeddings[start_idx].unsqueeze(1)
        end_node_embed = node_embeddings[end_idx].unsqueeze(1)
        waypoint_node_embeds = node_embeddings[waypoint_node_indices]
        pred = self.node_transformer_model(waypoint_node_embeds, start_node_embed, end_node_embed)

        return pred

model = GTTP()

def pairwise_ranking_loss(predicted_ordering, correct_ordering, margin=1.0):
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

import torch.nn as nn
import torch.optim as optim

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Define the number of epochs
num_epochs = 1000

# Move model to appropriate device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
graph.x = graph.x.to(device)
graph.edge_index = graph.edge_index.to(device)
graph.edge_attr = graph.edge_attr.to(device)

# Define a proper loss function
loss_fn = nn.MSELoss()

for batch in train_loader:
    break

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_loss = 0

    start_idx, waypoints_shuffled, waypoints_correct, end_idx = [x.to(device) for x in batch]

    # Forward pass
    predicted_ordering = model(
        graph.x, graph.edge_index, graph.edge_attr, start_idx, end_idx, waypoints_shuffled
    )

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

    # Print epoch loss
    avg_loss = total_loss
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")