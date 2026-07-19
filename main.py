import gymnasium as gym
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Arm_Net(nn.Module):
    def __init__(self, input_size, action_space_size):
        super().__init__()

        self.input = nn.Linear(input_size, 128)
        self.output = nn.Linear(128, action_space_size)

    def forward(self, input):
        l1 = F.relu(self.input(input))
        l2 = self.output(l1)
        return l2

def main():
    env = gym.make("Pusher", render_mode="human")
    observation, info = env.reset()
    episode_over = False

    while not episode_over:
        action = env.action_space.sample()
        action = np.zeros(7)
        action[0] = 1
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"observation: {observation}, reward: {reward}, info: {info}")
        time.sleep(.5)
        total_reward += reward
        episode_over = terminated or truncated

if __name__ == "__main__":
    main()
