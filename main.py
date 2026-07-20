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
        self.SD = nn.parameter(torch.zeros(action_space_size))

    def forward(self, input):
        l1 = F.relu(self.input(input))
        means = self.output(l1)
        return means, torch.exp(self.SD)

def main():
    env = gym.make("Pusher", render_mode="human")
    observation, info = env.reset()
    log_probabilities = []
    rewards = []
    episode_over = False
    net = Arm_Net(env.observation_space.shape[0], env.action_space.shape[0])

    while not episode_over:
        means, SD = net(observation)
        sample = torch.normal(means, SD)
        print(sample)

        observation, reward, terminated, truncated, info = env.step(action)
        print(f"observation: {observation}, reward: {reward}, info: {info}")
        time.sleep(.5)
        log_probabilities.append(torch.tensor(0.0)) 
        rewards.append(reward)
        episode_over = terminated or truncated

if __name__ == "__main__":
    main()
