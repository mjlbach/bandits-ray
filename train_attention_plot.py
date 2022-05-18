from pathlib import Path

import numpy as np
from ray.rllib.agents import ppo
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch

import numpy as np
from bandit.model import ComplexInputNetwork
from bandit.relational_env import env_creator, Edge

ModelCatalog.register_custom_model("graph_extractor", ComplexInputNetwork)

import torch
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

from torch_geometric.data import HeteroData, Batch

VIS_ENV_CONFIG = {
    "modalities": ["task_obs", "scene_graph"],
    "features": ["semantic_class"],
    "debug": False,
    "min_dummies": 75,
    "max_dummies": 75,
    "edge_groups": {
        "below": [
            Edge.below,
        ],
        "above": [
            Edge.above,
        ],
        "inRoom": [
            Edge.inRoom,
        ],
    }
}

def draw_attention(env, obs, color_dict):
    G = env.scene_graph.G
    pos = nx.get_node_attributes(G,'pos')
    label_dict = {i: int(category) for i, category in enumerate(obs["scene_graph"]["nodes"])}
    low, *_, high = sorted(color_dict.values())
    norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)
    nx.draw(G, 
            pos, 
            labels=label_dict, 
            with_labels = True,
            node_color=[mapper.to_rgba(i) for i in color_dict.values()])

def get_hetero_data_batch(obs):
    data = HeteroData()
    for key in obs["scene_graph"]:
        if key == 'nodes':
            data['node'].x = torch.tensor(obs['scene_graph']['nodes'], dtype=torch.float)
        else:
            data['node', key, 'node'].edge_index = torch.tensor(obs['scene_graph'][key], dtype=torch.long).T
    batch = Batch.from_data_list([data])
    return batch

def main(args):
    register_env("env_creator", env_creator)

    n_steps = 200
    num_envs = 8
    training_timesteps = 50000
    save_freq = 5000

    num_epochs = np.round(training_timesteps / n_steps).astype(int)
    save_ep_freq = np.round(num_epochs / (training_timesteps / save_freq)).astype(int)

    config = {
        "env": "env_creator",
        "model": {
            "custom_model": "graph_extractor",  # THIS LINE IS THE BROKEN ONE
            "custom_model_config": {
                "graph_model": args.model,
            },
            "post_fcnet_hiddens": [128, 128, 128],
            # "fcnet_hiddens": [128, 128, 128],
            "conv_filters": [[16, [4, 4], 4], [32, [4, 4], 4], [256, [8, 8], 2]],
        },
        "env_config": {
            "modalities": ["task_obs", "scene_graph"],
            "features": ["semantic_class"],
            "debug": False,
            "deploy_dummies": args.deploy_dummies,
            "min_dummies": args.min_dummies,
            "max_dummies": args.max_dummies,
            "edge_groups": {
                "below": [
                    Edge.below,
                ],
                "above": [
                    Edge.above,
                ],
                "inRoom": [
                    Edge.inRoom,
                ],
            }
        },
        "num_workers": num_envs,
        "framework": "torch",
        "seed": args.seed,
        "lr": 5e-5,
        "train_batch_size": n_steps,
        "num_gpus": 1,
    }

    experiment_save_path = "experiments"

    if args.name is None:
        experiment_name = "viz_{}_{}-{}_seed_{}".format(args.model, 
                                                    args.min_dummies, 
                                                    args.max_dummies,
                                                    args.seed)
    else:
        experiment_name = args.name

    experiment_path = Path(experiment_save_path, experiment_name)
    checkpoint_path = experiment_path.joinpath("checkpoints")
    log_path = experiment_path.joinpath("log")
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    Path(log_path).mkdir(parents=True, exist_ok=True)

    print(f"Saving to {checkpoint_path}")

    trainer = ppo.PPOTrainer(
        config,
        logger_creator=lambda x: UnifiedLogger(x, log_path),  # type: ignore
    )

    # define the sample env
    sample_env = env_creator(VIS_ENV_CONFIG)
    obs = sample_env.reset()
    num_nodes = sample_env.scene_graph.node_id_count
    batch = get_hetero_data_batch(obs)
    
    # draw the target attention map
    color_dict = {i: (1 if int(category) in [1, 2] else 0)  for i, category in enumerate(obs["scene_graph"]["nodes"])}
    draw_attention(sample_env, obs, color_dict)
    plt.savefig(experiment_path.joinpath("attention_target.png"))
    plt.clf()

    print("Model: ", args.model)
    print("Range of number of dummies: ", args.min_dummies, args.max_dummies)
    print("Seed: ", args.seed)

    for i in range(num_epochs):
        print("epoch:", i)
        trainer.train()

        # access the model that is being trained
        model = trainer.get_policy().model
        gnn_model = model.feature_extractors["scene_graph"]
        weights = gnn_model.get_weights(batch)
        weights = torch.flatten(weights).detach().numpy()
        
        # draw the attention map at each epoch
        color_dict = dict(zip(range(num_nodes), weights))
        draw_attention(sample_env, obs, color_dict)
        plt.savefig(experiment_path.joinpath("attention_ep_%d.png"%i))
        plt.clf()

        if (i % save_ep_freq) == 0:
            checkpoint = trainer.save(checkpoint_path)
            print("Checkpoint saved at", checkpoint)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        "-n",
        default=None,
        help="which config file to use [default: use yaml files in examples/configs]",
    )
    parser.add_argument(
        "--model",
        default="HGNN",
        help="GNN model",
    )
    parser.add_argument(
        "--deploy_dummies",
        type=bool,
        default=True,
        help="whether to deploy dummies",
    )
    parser.add_argument(
        "--min_dummies",
        type=int,
        default=0,
        help="minimum number of dummies",
    )
    parser.add_argument(
        "--max_dummies",
        type=int,
        default=0,
        help="maximum number of dummies",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed",
    )

    args = parser.parse_args()
    main(args)

# if __name__ == "__main__":
#     env = DebuggingEnv()
#     env.reset()
#     env.step(env.action_space.sample())
#
