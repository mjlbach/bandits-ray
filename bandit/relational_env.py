import gym
import numpy as np
from enum import Enum
from gym.spaces import Discrete, Box, Dict
from skimage.draw import disk, rectangle
from IPython import embed
import matplotlib.pyplot as plt
import networkx as nx

from ray.rllib.utils.spaces.repeated import Repeated

class Choice(Enum):
    left = 0
    right = 1


class Category(Enum):
    below = 0
    above = 1

class Edge(Enum):
    below = 0
    above = 1

class Graph:
    def __init__(self, 
                 env, 
                 features=["pos", "semantic_class"], 
                 edge_groups={"below": [Edge.below], "above": [Edge.above]}):
        self.env = env
        self.edge_type_to_group = {}
        self.edge_groups = edge_groups
        for group_label, edge_types in edge_groups.items():
            for edge in edge_types:
                self.edge_type_to_group[edge] = group_label

        self.edges_to_track = set()
        for edge_type in edge_groups.values():
            self.edges_to_track.update(edge_type)
        
        self.features = features
        self.feature_size_map = {
            "pos": 2,
            "semantic_class": 1
        }
        self.node_dim = 0
        for feature in features:
            self.node_dim += self.feature_size_map[feature]
            
        self.category_mapping = self.consolidate_mapping()
        
        self.reset()
        
    def consolidate_mapping(self):
        category_mapping = {}
        category_mapping["plane"] = 0
        category_mapping["object"] = 1
        return category_mapping
    
    def populate_graph(self):
        # planes
        self.left_center_id = self.get_node_id()
        self.G.add_node(
            self.left_center_id,
            pos=self.env.left_plane_center,
            semantic_class=self.category_mapping["plane"],
        )
        self.right_center_id = self.get_node_id()
        self.G.add_node(
            self.right_center_id,
            pos=self.env.right_plane_center,
            semantic_class=self.category_mapping["plane"],
        )

        # left object
        self.left_object_id = self.get_node_id()
        self.G.add_node(
            self.left_object_id,
            pos=self.env.left_object_center,
            semantic_class=self.category_mapping["object"],
        )
        self.G.add_edge(
            self.left_object_id,
            self.left_center_id,
            relation=Edge(self.env.object_position[0]),
        )

        # right object
        self.right_object_id = self.get_node_id()
        self.G.add_node(
            self.right_object_id,
            pos=self.env.right_object_center,
            semantic_class=self.category_mapping["object"],
        )
        self.G.add_edge(
            self.right_object_id,
            self.right_center_id,
            relation=Edge(self.env.object_position[1]),
        )

    def get_node_id(self):
        node_id_count = self.node_id_count
        self.node_id_count += 1
        return node_id_count

    def reset(self):
        self.G = nx.Graph()
        self.node_id_count = 0
        self.body_node_ids = {}
        
    def to_ray(self):
        edges = np.array(self.G.edges)
        nodes = np.zeros([len(self.G.nodes), self.node_dim], dtype=np.float32)
        for id in self.G.nodes:
            start = 0
            for feature in self.features:
                offset = self.feature_size_map[feature]
                nodes[id, start : start + offset] = self.G.nodes[id][feature]
                start += offset
        
        edge_groups = {group: [] for group in self.edge_groups}
        for (i, j, relation) in self.G.edges.data("relation"):  # type: ignore
            group = self.edge_type_to_group[relation]
            edge_groups[group].append((i, j))

        edges = {}
        for key in edge_groups:
            edges[key] = np.array(list(set(edge_groups[key])), dtype=np.int64)

        out = {"nodes": nodes}
        out.update(edges)
        return out


class RelationalEnv(gym.Env):
    def __init__(self, env_config):
        super().__init__()

        self.modalities = env_config["modalities"]
        self.action_space = Discrete(2)
        self.resolution = (128, 128, 3)
        obs_dict = {
            "task_obs": Box(low=0, high=1, shape=(2,)),
        }

        if "task_obs" in self.modalities:
            obs_dict["task_obs"] = Box(low=0, high=1, shape=(2,))

        if "rgb" in self.modalities:
            obs_dict["rgb"] = Box(
                low=0, high=255, shape=self.resolution, dtype=np.uint8
            )
        if "scene_graph" in self.modalities:
            self.scene_graph = Graph(
                self,
                edge_groups = env_config["edge_groups"],
                features = env_config["features"],
            )
            observation_dict = {
                    "nodes": Repeated(
                        Box(
                            low=-np.inf, high=np.inf, shape=(self.scene_graph.node_dim,), dtype=np.float32
                        ),
                        max_len=150,
                    ),
            }

            for edge_type in self.scene_graph.edge_groups:
                observation_dict[edge_type] = Repeated(
                    Box(low=0, high=1000, shape=(2,), dtype=np.int64),
                    max_len=300,
                )

            obs_dict["scene_graph"] = gym.spaces.Dict(observation_dict)

        self.left_plane_center = np.array(
            (self.resolution[0] / 2, self.resolution[1] / 4), 
            dtype=np.uint16,
        )
        self.right_plane_center = np.array(
            (self.resolution[0] / 2, self.resolution[1] - self.resolution[1] / 4),
            dtype=np.uint16,
        )

        self.left_object_center = None
        self.right_object_center = None

        self.observation_space = Dict(obs_dict)

        self.debug = env_config.get("debug", False)
        self.rng = np.random.default_rng()

    def observe(self):
        obs = dict()

        state = self.object_position
        if "task_obs" in self.modalities:
            obs["task_obs"] = state.astype(np.float32)

        goal = np.zeros((2,))
        goal[self.target_obj_category] = 1
        obs["task_obs"] = goal.astype(np.float32)

        if "rgb" in self.modalities:
            img = np.zeros(self.resolution, dtype=np.uint8)
            
            for plane_center in (self.left_plane_center, self.right_plane_center):
                plane_extent = np.array([5, (self.resolution[1]/2-4)], dtype=np.uint16)
                plane_start = plane_center - (plane_extent/2)
                plane_start = plane_start.astype(np.uint16)
                rr, cc = rectangle(plane_start, extent=plane_extent, shape=img.shape)
                img[rr, cc, :] = np.array([0, 255, 0], dtype=np.uint8)

            for object_center in (self.left_object_center, self.right_object_center):
                rr, cc = disk(object_center, 10, shape=img.shape)
                img[rr, cc, :] = np.array([255, 0, 0], dtype=np.uint8)

            obs["rgb"] = img

        if "scene_graph" in self.modalities:
            obs["scene_graph"] = self.scene_graph.to_ray()

        return obs

    def get_object_center(self, object, plane_center):
        if Category(object) is Category.below:
            return plane_center + self.rng.integers(
                low=(self.resolution[0] // 8, -self.resolution[1] // 6),
                high=(self.resolution[0] // 4, self.resolution[1] // 6),
            )
        elif Category(object) is Category.above:
            return plane_center + self.rng.integers(
                low=(-self.resolution[0] // 4, -self.resolution[1] // 6),
                high=(-self.resolution[0] // 8, self.resolution[1] // 6),
            )

    def reset(self):
        # Sample objects on left and right
        self.object_position = self.rng.choice(len(Category), 2, replace=False)  # type: ignore

        # Choose below or above as goal
        target_choice = self.rng.choice(self.object_position)  # type: ignore

        self.target_obj_category = self.object_position[target_choice]

        self.left_object_center = self.get_object_center(self.object_position[0],
                                                          self.left_plane_center)
        self.right_object_center = self.get_object_center(self.object_position[1],
                                                           self.right_plane_center)
        
        if "scene_graph" in self.modalities:
            self.scene_graph.reset()
            self.scene_graph.populate_graph()

        return self.observe()

    def step(self, action):

        if self.object_position[action] == self.target_obj_category:
            reward = 1
        else:
            reward = 0

        obs = self.observe()

        if self.debug:
            print()
            print("Episode:")
            print("-" * 30)
            print(f"Goal: {Category(self.target_obj_category)}")
            print(
                f"State: Left: {Category(self.object_position[0])}, Right: {Category(self.object_position[1])}"
            )
            print(f"Action: {Choice(action)}")
            print(f"Reward: {reward}")
            print("-" * 30)

        return (obs, reward, True, {})


def env_creator(env_config):
    return RelationalEnv(env_config)


if __name__ == "__main__":
    env = env_creator("")
    obs = env.reset()
    breakpoint()
    env.observation_space.contains(obs)
    import torch
    from torch_geometric.data import HeteroData, Batch
    from bandit.models.hetero.gnn import HGNN

    # data = Data(x=torch.tensor(obs['scene_graph']['nodes'], dtype=torch.float), edge_index=torch.tensor(obs['scene_graph']['edges'], dtype=torch.long))
    data = HeteroData()
    for key in obs["scene_graph"]:
        if key == 'nodes':
            data['node'].x = torch.tensor(obs['scene_graph']['nodes'], dtype=torch.float)
        else:
            data['node', key, 'node'].edge_index = torch.tensor(obs['scene_graph'][key], dtype=torch.long).T
    batch = Batch.from_data_list([data])

    with torch.no_grad():
        model = HGNN(in_features=data['node'].x.shape[1], metadata=data.metadata())
        out = model(batch)

    out = model(batch)
