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
        time.sleep(.001)
    return log_probabilities, rewards

def accumulate_rewards(rewards):
    accumulated_rewards = []
    for i in range(len(rewards)):
        slice = rewards[i:]
        reward = 0
        for j in range(len(slice)):
            discount = 0.99**j
            reward += slice[j] * discount
        accumulated_rewards.append(reward)
    return accumulated_rewards

def calculate_mean_reward_for_timestep(episodes: list) -> list:
    returns = [accumulate_rewards(e) for e in episodes]
    mean_reward_for_timestep = []
    len_of_episode = len(returns[0])
    for i in range(len_of_episode):
        value = 0 
        for episode in returns:
            value += episode[i]
        mean = torch.tensor(value / len(returns))
        mean_reward_for_timestep.append(mean)
    return mean_reward_for_timestep

def main():
    env = gym.make("Pusher", render_mode="human")
    net = Arm_Net(env.observation_space.shape[0], env.action_space.shape[0])
    optimizer = torch.optim.Adam(net.parameters())
    episode_length = 50
    for i in range(1000):
        print(f"Generation {i}")
        log_probabilities_episodes, rewards_episodes = [], []
        for j in range(episode_length):
            log_probabilities, rewards = run_episode(net=net, env=env)
            log_probabilities_episodes.append(log_probabilities)
            rewards_episodes.append(rewards)
        mean_reward_for_timestep = torch.stack(calculate_mean_reward_for_timestep(rewards_episodes))
        loss = 0
        for j in range(episode_length):
            returns = torch.stack(accumulate_rewards(rewards_episodes[j]))
            advantage = returns - mean_reward_for_timestep.detach()
            loss = loss - (torch.stack(log_probabilities_episodes[j]) * advantage).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    env.close()

if __name__ == "__main__":
    main()
