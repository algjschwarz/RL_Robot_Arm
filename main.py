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
        self.SD = nn.Parameter(torch.zeros(action_space_size))

    def forward(self, input):
        l1 = F.relu(self.input(input))
        means = self.output(l1)
        return means, torch.exp(self.SD)

def run_episode(net, env) -> list:
    observation, info = env.reset()
    log_probabilities = []
    rewards = []
    episode_over = False
    while not episode_over:
        observation = torch.tensor(observation, dtype=torch.float32)
        means, SD = net(observation)
        dist = torch.distributions.Normal(means, SD)
        sample = dist.sample()
        log_prob = dist.log_prob(sample)
        action = sample.detach().numpy()

        observation, reward, terminated, truncated, info = env.step(action)
        log_probabilities.append(log_prob.sum()) 
        rewards.append(torch.tensor(reward))
        episode_over = terminated or truncated
        time.sleep(.01)
    return log_probabilities, rewards

def accumulate_rewards(rewards):
    accumulated_rewards = []
    for i in range(len(rewards)):
        accumulated_rewards.append(sum(rewards[i:]))
    return accumulated_rewards

def update_model_weights(log_probabilities: list, rewards: list, optimizer, net):
    accumulated_rewards = torch.stack(accumulate_rewards(rewards))
    accumulated_rewards = accumulated_rewards - accumulated_rewards.mean()
    loss = -(torch.stack(log_probabilities) * accumulated_rewards).sum()
    net.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    env = gym.make("Pusher", render_mode="human")
    net = Arm_Net(env.observation_space.shape[0], env.action_space.shape[0])
    optimizer = torch.optim.Adam(net.parameters())
    for i in range(1000):
        print(f"Generation {i}")
        log_probabilities, rewards = run_episode(net=net, env=env)
        update_model_weights(log_probabilities=log_probabilities, rewards=rewards, optimizer=optimizer, net=net)

    env.close()

if __name__ == "__main__":
    main()
