import gym
import numpy as np
from enum import Enum
from gym.spaces import Discrete, Box, Dict

class Choice(Enum):
    left = 0
    right = 1

class Category(Enum):
    square = 0
    circle = 1

class DebuggingEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.action_space = Discrete(2)
        self.observation_space = Dict({
            'task_obs': Box(low=0, high=1, shape=(2,)),
            'vectorized_goal': Box(low=0, high=1, shape=(2,)),
        })

        self.debug = True
        self.rng = np.random.default_rng()

    def observe(self):
        obs = dict()

        state = self.object_position
        obs["task_obs"] = state.astype(np.float32)

        goal = np.zeros((2,))
        goal[self.target_obj_category] = 1
        obs["vectorized_goal"] = goal.astype(np.float32)
        return obs

    def reset(self):
        # Sample objects on left and right
        self.object_position = self.rng.choice(len(Category), 2, replace=False) #type: ignore

        # Choose left or right as goal
        target_choice = self.rng.choice(self.object_position) #type: ignore

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
            print(f"State: Left: {Category(self.object_position[0])}, Right: {Category(self.object_position[1])}")
            print(f"Action: {Choice(action)}")
            print(f"Reward: {reward}")
            print("-" * 30)

        return(obs, reward, True, {})

def env_creator(_):
    return DebuggingEnv() 

