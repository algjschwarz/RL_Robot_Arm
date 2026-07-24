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

class Critic(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input = nn.Linear(input_size, 32)
        self.output = nn.Linear(32, 1)
    def forward(self, input):
        l1 = F.relu(self.input(input))
        return self.output(l1)

def run_episode(net, env, observation) -> list:
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
        #time.sleep(.001)
    return log_probabilities, rewards

def accumulate_rewards(rewards, gamma=0.95):
    accumulated_rewards = [0] * len(rewards)
    running = 0
    for i in reversed(range(len(rewards))):
        running = rewards[i] + gamma * running
        accumulated_rewards[i] = running
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

def run_episodes(episode_length, env, watch_env, net, critic, generation) -> list:
    log_probabilities_episodes, rewards_episodes, predicted_returns = [], [], []
    for j in range(episode_length):
        observation, info = env.reset()
        obs = torch.tensor(observation, dtype=torch.float32)
        predicted_returns.append(critic(obs))
        if 2950 < generation < 3000:
            environment = watch_env
        else:
            environment = env
        log_probabilities, rewards = run_episode(net=net, env=environment, observation=observation)
        log_probabilities_episodes.append(log_probabilities)
        rewards_episodes.append(rewards)
    return log_probabilities_episodes, rewards_episodes, predicted_returns

def main():
    env = gym.make("Pusher")
    watch_env = gym.make("Pusher", render_mode="human")
    net = Arm_Net(env.observation_space.shape[0], env.action_space.shape[0])
    critic = Critic(env.observation_space.shape[0])
    optimizer = torch.optim.Adam(net.parameters())
    critic_optimizer = torch.optim.Adam(critic.parameters())
    episode_length = 10
    lowest_average = -500
    generation_of_lowest_average = 0
    for i in range(1000):
        log_probabilities_episodes, rewards_episodes, predicted_returns = run_episodes(episode_length, env, watch_env, net, critic, generation=i)
        #mean_reward_for_timestep = torch.stack(calculate_mean_reward_for_timestep(rewards_episodes))
        loss = 0
        critic_loss = 0
        for j in range(episode_length):
            returns = torch.stack(accumulate_rewards(rewards_episodes[j]))[0]
            expected_return = predicted_returns[j]
            advantage = returns - expected_return.detach()
            loss = loss - (torch.stack(log_probabilities_episodes[j]) * advantage).sum()
            critic_loss = critic_loss + (expected_return - returns)**2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        avg_total = sum(sum(r).item() for r in rewards_episodes) / episode_length
        if avg_total > lowest_average:
            generation_of_lowest_average = i
            lowest_average = avg_total

        print(f"Generation {i}  avg reward {avg_total:.1f}  Critic Loss {critic_loss}")
    print(f"Highest Average {lowest_average}, Generation of highest average {generation_of_lowest_average}")
    env.close()
    watch_env.close()

if __name__ == "__main__":
    main()
