import re
from pathlib import Path

import numpy as np
from ray.rllib.agents import ppo
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env
import os
from ray.rllib.models.catalog import ModelCatalog

import numpy as np
from bandit.model import ComplexInputNetwork
from bandit.env import env_creator

ModelCatalog.register_custom_model("graph_extractor", ComplexInputNetwork)

def main(args):
    register_env("env_creator", env_creator)

    n_steps = 160
    num_envs = 8

    experiment_save_path = os.path.expanduser("~/ray_results")
    experiment_name = args.name
    training_timesteps = 5e8
    save_freq = 1e6

    checkpoint_path = Path(experiment_save_path, experiment_name)
    num_epochs = np.round(training_timesteps / n_steps).astype(int)
    save_ep_freq = np.round(
        num_epochs / (training_timesteps / save_freq)
    ).astype(int)

    config = {
        "env": "env_creator",
        "model": {
          "custom_model": "graph_extractor", # THIS LINE IS THE BROKEN ONE
          # "post_fcnet_hiddens": [128, 128, 128],
          # "fcnet_hiddens": [128, 128, 128],
          "conv_filters": [[16, [4, 4], 4], [32, [4, 4], 4], [256, [8, 8], 2]]
        },
        "num_workers":num_envs,
        "framework": "torch",
        "seed": 0,
        "lambda": 0.9,
        "lr": 1e-4,
        "train_batch_size": n_steps,
        "rollout_fragment_length":  n_steps // num_envs,
        "num_sgd_iter": 30,
        "sgd_minibatch_size": 128,
        "gamma": 0.99,
        "create_env_on_driver": False,
        "num_gpus": 0,
        # "callbacks": MetricsCallback,
        # "log_level": "DEBUG",
        # "_disable_preprocessor_api": False,
    }

    log_path = str(checkpoint_path.joinpath("log"))
    print(f"Saving to {log_path}")
    Path(log_path).mkdir(parents=True, exist_ok=True)
    trainer = ppo.PPOTrainer(
        config,
        logger_creator=lambda x: UnifiedLogger(x, log_path), #type: ignore
    )

    if Path(checkpoint_path).exists():
        checkpoints = Path(checkpoint_path).rglob("checkpoint-*")
        checkpoints = [
            str(f) for f in checkpoints if re.search(r".*checkpoint-\d*$", str(f))
        ]
        checkpoints = sorted(checkpoints)
        if len(checkpoints) > 0:
            trainer.restore(checkpoints[-1])

    for i in range(num_epochs):
        # Perform one iteration of training the policy with PPO
        trainer.train()

        if (i % save_ep_freq) == 0:
            checkpoint = trainer.save(checkpoint_path)
            print("checkpoint saved at", checkpoint)


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
