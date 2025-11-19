#!/usr/bin/env python3

import gym
from ns3gym import ns3env
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Configuration
port = 5555
simulation_time = 100  # seconds
step_time = 1.0

# Create environment
env = ns3env.Ns3Env(port=port, stepTime=step_time, startSim=True,
                    simSeed=0, simArgs={}, debug=False)

# Check environment
check_env(env)

# Create PPO model
model = PPO("MlpPolicy", env, verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10)

# Train the model
print("Starting training...")
model.learn(total_timesteps=10000)

# Save the model
model.save("gemini_ppo_model")

# Test the trained model
print("Testing trained model...")
obs = env.reset()
total_reward = 0
episode_count = 0

while episode_count < 10:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward

    if done:
        print(f"Episode {episode_count + 1}: Total Reward = {total_reward}")
        total_reward = 0
        episode_count += 1
        obs = env.reset()

env.close()
