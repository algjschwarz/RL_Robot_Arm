import gymnasium as gym
import time

env = gym.make("Pusher", render_mode="human")

observation, info = env.reset()

print(observation)

episode_over = False
total_reward = 0

while not episode_over:
    print(env.action_space)
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    time.sleep(.5)
    total_reward += reward
    episode_over = terminated or truncated

print(total_reward)
env.close()
