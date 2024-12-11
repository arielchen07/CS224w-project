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
from mapUtil import CityMap, createMap
from searchUtil import ShortestPathProblem, UniformCostSearch
from calculatePath import extractPath
# from plotMap import getTotalCost

# Set the random seed for reproducibility
seed = 42  # Replace with your desired seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Enable deterministic mode
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

R = 6371000  

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

def getTotalCost(path: List[str], cityMap: CityMap) -> float:
    """
    Return the length of a given path
    """
    cost = 0.0
    for i in range(len(path) - 1):
        print(cityMap.distances[path[i]][path[i + 1]])
        cost += cityMap.distances[path[i]][path[i + 1]]
    return cost


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
            self.node_idx_to_id = {}
            self.idx_to_node_id = {}
            self.id_counter = 0

            self.edge_dict = {}

            self.anchor_points_idxs = []
        
        def generate_anchor_points(self):
            self.anchor_points_idxs = random.sample(range(len(self.nodes)), 100)

        def node(self, n: osmium.osm.Node) -> None:
            # if len(self.nodes) < 1000:
            if True:
                xy = xy_distance(n.location.lat, n.location.lon, anchor_point[0], anchor_point[1])
                self.nodes.append([xy[0], xy[1]])
                self.node_id_to_idx[n.id] = self.id_counter
                self.node_idx_to_id[self.id_counter] = n.id
                self.idx_to_node_id[self.id_counter] = n.id
                self.id_counter += 1

        def way(self, w):
            node_refs = [node.ref for node in w.nodes]

            for i in range(len(node_refs) - 1):
                node_start = node_refs[i]
                node_end = node_refs[i + 1]

                if node_start in self.node_id_to_idx and node_end in self.node_id_to_idx:

                    node_1_idx = self.node_id_to_idx[node_start]
                    node_2_idx = self.node_id_to_idx[node_end]

                    if (node_1_idx, node_2_idx) not in self.edge_dict:
                        self.edge_dict[(node_1_idx, node_2_idx)] = True
                        self.edge_dict[(node_2_idx, node_1_idx)] = True
                    
                        self.edges[0].append(node_1_idx)
                        self.edges[1].append(node_2_idx)

                        if node_1_idx < len(self.nodes) and node_2_idx < len(self.nodes):

                            node_1 = self.nodes[node_1_idx]
                            node_2 = self.nodes[node_2_idx]

                            n1_lat, n1_lon = node_1
                            n2_lat, n2_lon = node_2

                            dist = compute_distance(n1_lat, n1_lon, n2_lat, n2_lon)
                            self.edge_dist.append(dist)

    mapCreator = MapCreationHandler()
    mapCreator.apply_file(osmPath, locations=True)

    mapCreator.generate_anchor_points()
    print(len(mapCreator.anchor_points_idxs))

    stanfordCalMap = createMap("./data/stanford.pbf")
    usc = UniformCostSearch(verbose=0)

    for i in range(len(mapCreator.nodes)):
        currStart = mapCreator.idx_to_node_id[i]
        node_ = []
        for idx in mapCreator.anchor_points_idxs:
            currEnd = mapCreator.idx_to_node_id[idx]
            problem = ShortestPathProblem(startLocation=currStart, endTag=currEnd, cityMap=stanfordCalMap)
            usc.solve(problem)
            currPath = extractPath(problem.startLocation, usc)
            print("currPath", currPath)
            
            # Here is an example
            currPathCost = getTotalCost(currPath, stanfordCalMap)
            print("currPathCost", currPathCost)

            # xy = xy_distance(n.location.lat, n.location.lon, anchor_point[0], anchor_point[1])
            node_.append(currPathCost)
        
        mapCreator.nodes.append(node_)

    x = torch.tensor(mapCreator.nodes, dtype=torch.float)
    edge_index = torch.tensor(mapCreator.edges, dtype=torch.long)
    edge_attr = torch.tensor(mapCreator.edge_dist, dtype=torch.float).unsqueeze(1)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data, mapCreator.node_id_to_idx


class PathDataset(Dataset):

    def __init__(self, route_files: List[str], node_id_to_idx: dict, fixed_length: int = 6):
        self.route_files = route_files
        self.node_id_to_idx = node_id_to_idx
        self.fixed_length = fixed_length
        self.data = self._load_data()

    def _load_data(self):
        samples = []
        for f in self.route_files:
            with open(f, 'r') as json_file:
                route_info = json.load(json_file)

            start_id = int(route_info["start"])
            end_id = int(route_info["end"])
            waypoint_tags = route_info["waypointTags"]
            waypoint_ids = [int(tag.split('=')[1]) for tag in waypoint_tags]

            try:
                start_idx = self.node_id_to_idx[start_id]
                end_idx = self.node_id_to_idx[end_id]
                waypoints_correct = [self.node_id_to_idx[w_id] for w_id in waypoint_ids]

                if len(waypoints_correct) != 6:
                    continue

            except KeyError as e:
                print(f"KeyError: {e}")
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

        waypoints_shuffled = waypoints_correct[:]
        
        # Randomly shuffle waypoints
        random.shuffle(waypoints_shuffled)

        # Create a mapping of shuffled indices
        shuffled_indices = [waypoints_correct.index(wp) for wp in waypoints_shuffled]

        # Pad way points if length is different
        if len(waypoints_shuffled) < self.fixed_length:
            padding_needed = self.fixed_length - len(waypoints_shuffled)
            waypoints_shuffled = waypoints_shuffled + [-1] * padding_needed
            shuffled_indices = shuffled_indices + [-1] * padding_needed

        waypoints_shuffled_tensor = torch.tensor(waypoints_shuffled, dtype=torch.long)
        shuffled_order_tensor = torch.tensor(shuffled_indices, dtype=torch.long)

        return start_idx, waypoints_shuffled_tensor, shuffled_order_tensor, end_idx
    

def load_path_data(route_dir: str, node_id_to_idx: dict, train_ratio=0.8, val_ratio=0.1, seed=42):

    torch.manual_seed(seed)

    route_files = [os.path.join(route_dir, f) for f in os.listdir(route_dir) if f.endswith('.json')]
    
    full_dataset = PathDataset(route_files, node_id_to_idx)
    dataset_len = len(full_dataset)
    train_len = int(dataset_len * train_ratio)
    val_len = int(dataset_len * val_ratio)
    test_len = dataset_len - train_len - val_len

    train_dataset_raw, val_dataset_raw, test_dataset_raw = random_split(full_dataset, [train_len, val_len, test_len])

    train_route_files = [route_files[i] for i in train_dataset_raw.indices]
    val_route_files = [route_files[i] for i in val_dataset_raw.indices]
    test_route_files = [route_files[i] for i in test_dataset_raw.indices]

    train_dataset = PathDataset(train_route_files, node_id_to_idx)
    val_dataset = PathDataset(val_route_files, node_id_to_idx)
    test_dataset = PathDataset(test_route_files, node_id_to_idx)

    return train_dataset, val_dataset, test_dataset


def normalize_features(features, a=0.0, b=1.0):
    min_val = features.min()
    max_val = features.max()

    if max_val == min_val:
        # All edge_attr values are identical
        return torch.full_like(features, (a + b) / 2)  # Set to the midpoint of the range

    # Min-max normalization to range [a, b]
    normalized_features = a + (features - min_val) * (b - a) / (max_val - min_val)
    return normalized_features


def run_evaluate(model, graph, loader, device="cuda"):
    model.eval()

    total_loss = 0
    total_num = 0
    correct_num = 0
    num_samples = 0

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
                    total_loss += loss.item() * start_idx.size(0)
                    num_samples += start_idx.size(0)

        # Print epoch loss
        avg_loss = total_loss / num_samples
        avg_acc = correct_num / total_num
        print(f"Validation Loss: {avg_loss:.4f}")
        print(f"Validation Accuracy: {avg_acc:.4f}")

        return avg_loss, avg_acc


def set_deterministic(seed=42):
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Enable deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def construct_small_graph(mode):
    nodes = []
    for i in range(10):
        for j in range(10):

            if mode in ["same_embedding", "gnn_embedding", "pretrain_gnn_embedding"]:
                nodes.append([random.random() * 10, random.random() * 10])
            elif mode == "coord_feature":
                nodes.append([i, j])
            else:
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

    node_id_to_idx = {}

    for i in range(10):
        for j in range(10):
            node_id_to_idx[f"{i},{j}"] = i*10+j

    return graph, node_id_to_idx


def get_adj_matrix(graph):
    adj_matrix = torch.zeros((len(graph.x), len(graph.x)), dtype=torch.float32)
    adj_matrix[graph.edge_index[0], graph.edge_index[1]] = 1.0

    return adj_matrix