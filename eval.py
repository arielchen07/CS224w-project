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

from searchUtil import ShortestPathProblem, UniformCostSearch
from mapUtil import CityMap, createMap
from calculatePath import extractPath
from utils import getTotalCost

from utils import load_path_data, normalize_features, set_deterministic, construct_small_graph, run_evaluate, construct_graph
from model import GTTP

def sample_path(graph, model, start_idx, end_idx, waypoints):

    waypoint_list = waypoints.tolist()
    ordering = {waypoint: {'before': set(), 'after': set()} for waypoint in waypoint_list}

    for i in range(len(waypoint_list)):
        for j in range(i + 1, len(waypoint_list)):

            pred = model(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                start_idx.unsqueeze(0),
                end_idx.unsqueeze(0),
                waypoints[i].unsqueeze(0), waypoints[j].unsqueeze(0))

            pred = torch.sigmoid(pred).squeeze(1).item()

            if pred > 0.5:
                ordering[waypoint_list[i]]['after'].add(waypoint_list[j])
                ordering[waypoint_list[j]]['before'].add(waypoint_list[i])
            else:
                ordering[waypoint_list[j]]['after'].add(waypoint_list[i])
                ordering[waypoint_list[i]]['before'].add(waypoint_list[j])
    
    # Greedy ordering algorithm
    ordered = []
    remaining = set(ordering)
    
    while remaining:

        chosen = min(
            remaining, 
            key=lambda point: len(ordering[point]['before'].intersection(remaining))
        )
        
        ordered.append(chosen)
        remaining.remove(chosen)

    return ordered

if __name__ == "__main__":

    set_deterministic(100)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mode = "gnn_embedding"
    model_file = f"trained_models/{mode}.pt"

    graph_path = "data/stanford.pbf"
    graph_fname = graph_path.split("/")[-1].split(".")[0]

    map = createMap(graph_path)

    with open(f"graph_cache/{graph_fname}_0.pkl", "rb") as f:
        graph, node_id_to_idx = pickle.load(f)

    idx_to_node_id = {v: k for k, v in node_id_to_idx.items()}

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

    embed_dim = 128
    gtn_hidden_dim = 32
    input_dim = 2

    model = GTTP(input_dim, gtn_hidden_dim, embed_dim, gtn_layers=3, dropout=0.1)
    model = model.to(device)
    model.eval()

    model.load_state_dict(torch.load(model_file))

    # train_avg_loss, train_avg_acc = run_evaluate(model, graph, train_loader, device)
    # val_avg_loss, val_avg_acc = run_evaluate(model, graph, val_loader, device)

    # print(f"val_avg_loss: {val_avg_loss}, val_avg_acc: {val_avg_acc}")
    # print(f"train_avg_loss: {train_avg_loss}, train_avg_acc: {train_avg_acc}")

    def distance_between_two_nodes(map, start, end):
        problem = ShortestPathProblem(startLocation=str(start), endTag=f"label={end}", cityMap=map)
        usc = UniformCostSearch(verbose=0)
        usc.solve(problem)
        path = extractPath(problem.startLocation, usc)
        cost = getTotalCost(path, map)
        return cost

    pred_distance_list = []
    correct_distance_list = []
    random_distance_list = []

    for batch in train_loader:
        
        start_idx, waypoints_shuffled, waypoints_correct, end_idx = [x.to(device) for x in batch]

        for i in range(len(start_idx)):

            try:
                path = sample_path(graph, model, start_idx[i], end_idx[i], waypoints_shuffled[i])
                waypoints_shuffled_indices = waypoints_shuffled[i].tolist()
                waypoints_correct_ordering = waypoints_correct[i].tolist()

                random_path = waypoints_shuffled_indices.copy()

                correct_path = []
                for k in range(len(waypoints_shuffled_indices)):
                    correct_path.append(waypoints_shuffled_indices[waypoints_correct_ordering.index(k)])

                start_idx_item = start_idx[i].item()
                end_idx_item = end_idx[i].item()

                start_node_id = idx_to_node_id[start_idx_item]
                end_node_id = idx_to_node_id[end_idx_item]

                path = [start_node_id] + [idx_to_node_id[node] for node in path] + [end_node_id]
                correct_path = [start_node_id] + [idx_to_node_id[node] for node in correct_path] + [end_node_id]
                random_path = [start_node_id] + [idx_to_node_id[node] for node in random_path] + [end_node_id]

                pred_distance = 0
                correct_distance = 0
                random_distance = 0

                for i in range(len(path) - 1):
                    
                    pred_path_start = path[i]
                    pred_path_end = path[i + 1]

                    correct_path_start = correct_path[i]
                    correct_path_end = correct_path[i + 1]

                    random_path_start = random_path[i]
                    random_path_end = random_path[i + 1]

                    pred_distance += distance_between_two_nodes(map, pred_path_start, pred_path_end)
                    correct_distance += distance_between_two_nodes(map, correct_path_start, correct_path_end)
                    random_distance += distance_between_two_nodes(map, random_path_start, random_path_end)

                print(f"Predicted Path: {path} Distance: {pred_distance}")
                print(f"Correct Path: {correct_path} Distance: {correct_distance}")
                print(f"Random Path: {random_path} Distance: {random_distance}")

                pred_distance_list.append(pred_distance)
                correct_distance_list.append(correct_distance)
                random_distance_list.append(random_distance)
            except:
                pass
    
    print(f"pred_distance_list: {pred_distance_list}")
    print(f"correct_distance_list: {correct_distance_list}")
    print(f"random_distance_list: {random_distance_list}")
    print(f"Average ratio: {np.mean(np.array(pred_distance_list) / np.array(correct_distance_list))}")
    print(f"Average random ratio: {np.mean(np.array(random_distance_list) / np.array(correct_distance_list))}")