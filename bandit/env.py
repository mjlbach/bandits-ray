import gym
import numpy as np
from enum import Enum
from gym.spaces import Discrete, Box, Dict
from skimage.draw import disk, rectangle
from IPython import embed
import matplotlib.pyplot as plt


class Choice(Enum):
    left = 0
    right = 1


class Category(Enum):
    square = 0
    circle = 1


class DebuggingEnv(gym.Env):
    def __init__(self, debug=False, modalities=["rgb", "vectorized_goal"]):
        super().__init__()

        self.modalities = modalities
        self.action_space = Discrete(2)
        self.resolution = (128, 128, 3)
        obs_dict = {
            "vectorized_goal": Box(low=0, high=1, shape=(2,)),
        }

        if "task_obs" in self.modalities:
            obs_dict["task_obs"] = Box(low=0, high=1, shape=(2,))

        if "rgb" in self.modalities:
            obs_dict["rgb"] = Box(
                low=0, high=255, shape=self.resolution, dtype=np.uint8
            )
        self.observation_space = Dict(obs_dict)

        self.debug = debug
        self.rng = np.random.default_rng()

    def observe(self):
        obs = dict()

        state = self.object_position
        if "task_obs" in self.modalities:
            obs["task_obs"] = state.astype(np.float32)

        goal = np.zeros((2,))
        goal[self.target_obj_category] = 1
        obs["vectorized_goal"] = goal.astype(np.float32)

        if "rgb" in self.modalities:
            left_start = np.array(
                (self.resolution[0] / 2, self.resolution[1] / 4), dtype=np.uint16
            )
            right_start = np.array(
                (self.resolution[0] // 2, self.resolution[1] - self.resolution[1] // 4),
                dtype=np.uint16,
            )

            img = np.zeros(self.resolution, dtype=np.uint8)

            for element, center in zip(self.object_position, (left_start, right_start)):
                if Category(element) is Category.circle:
                    rr, cc = disk(center, 10, shape=img.shape)
                    img[rr, cc, :] = np.array([255, 0, 0], dtype=np.uint8)
                elif Category(element) is Category.square:
                    extent = (30, 30)
                    rr, cc = rectangle(center, extent=extent, shape=img.shape)
                    img[rr, cc, :] = np.array([255, 0, 0], dtype=np.uint8)
            obs["rgb"] = img

        return obs

    def reset(self):
        # Sample objects on left and right
        self.object_position = self.rng.choice(len(Category), 2, replace=False)  # type: ignore

        # Choose left or right as goal
        target_choice = self.rng.choice(self.object_position)  # type: ignore

        self.target_obj_category = self.object_position[target_choice]
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


def env_creator(_):
    return DebuggingEnv()


if __name__ == "__main__":
    env = env_creator("")
    obs = env.reset()
