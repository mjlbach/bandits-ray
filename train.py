from pathlib import Path

import numpy as np
from ray.rllib.agents import ppo
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env
from ray.rllib.models.catalog import ModelCatalog

import numpy as np
from bandit.model import ComplexInputNetwork
from bandit.relational_env import env_creator, Edge

ModelCatalog.register_custom_model("graph_extractor", ComplexInputNetwork)


def main(args):
    register_env("env_creator", env_creator)

    n_steps = 160
    num_envs = 8

    training_timesteps = 5e8
    save_freq = 1e6

    num_epochs = np.round(training_timesteps / n_steps).astype(int)
    save_ep_freq = np.round(num_epochs / (training_timesteps / save_freq)).astype(int)

    config = {
        "env": "env_creator",
        "model": {
            "custom_model": "graph_extractor",  # THIS LINE IS THE BROKEN ONE
            "custom_model_config": {
                "graph_model": "HGNN",
            },
            "post_fcnet_hiddens": [128, 128, 128],
            # "fcnet_hiddens": [128, 128, 128],
            "conv_filters": [[16, [4, 4], 4], [32, [4, 4], 4], [256, [8, 8], 2]],
        },
        "env_config": {
            "modalities": ["task_obs", "scene_graph"],
            "features": ["pos", "semantic_class"],
            "debug": False,
            # "edge_groups": {
            #     "edges": [
            #         Edge.below,
            #         Edge.above,
            #         Edge.inRoom,
            #     ],
            # }
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
        "seed": 0,
        # "lambda": 0.9,
        "lr": 1e-4,
        # "train_batch_size": n_steps,
        # "rollout_fragment_length":  n_steps // num_envs,
        # "num_sgd_iter": 30,
        # "sgd_minibatch_size": 128,
        # "gamma": 0.99,
        # "create_env_on_driver": False,
        "num_gpus": 1,
        # "callbacks": MetricsCallback,
        # "log_level": "DEBUG",
        # "_disable_preprocessor_api": False,
    }

    experiment_save_path = "experiments"
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

    for i in range(num_epochs):
        trainer.train()
        if (i % save_ep_freq) == 0:
            checkpoint = trainer.save(checkpoint_path)
            print("Checkpoint saved at", checkpoint)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        "-n",
        default="test",
        help="which config file to use [default: use yaml files in examples/configs]",
    )
    args = parser.parse_args()
    main(args)

# if __name__ == "__main__":
#     env = DebuggingEnv()
#     env.reset()
#     env.step(env.action_space.sample())
#
