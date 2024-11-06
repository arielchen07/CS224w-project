import osmium
from osmium import osm

import torch
from torch_geometric.data import Data

def construct_graph(osmPath: str):
    class MapCreationHandler(osmium.SimpleHandler):
        def __init__(self) -> None:
            super().__init__()
            self.nodes = []
            self.edges = [[], []]

            self.node_id_to_idx = {}
            self.idx_to_node_id = {}
            self.id_counter = 0

        def node(self, n: osm.Node) -> None:
            self.nodes.append([n.location.lat, n.location.lon])
            self.node_id_to_idx[n.id] = self.id_counter
            self.idx_to_node_id[self.id_counter] = n.id
            self.id_counter += 1

        def way(self, w):
            node_refs = [node.ref for node in w.nodes]

            for i in range(len(node_refs) - 1):
                node_start = node_refs[i]
                node_end = node_refs[i + 1]
                
                self.edges[0].append(self.node_id_to_idx[node_start])
                self.edges[1].append(self.node_id_to_idx[node_end])

    mapCreator = MapCreationHandler()
    mapCreator.apply_file(osmPath, locations=True)

    x = torch.tensor(mapCreator.nodes, dtype=torch.float)
    edge_index = torch.tensor(mapCreator.edges, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    print(data)

if __name__ == "__main__":
    construct_graph("stanford.pbf")
