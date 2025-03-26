import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from humanoid_v4 import HumanoidEnv
from humanoidstandup_v4 import HumanoidStandupEnv #do later 


gym.pprint_registry()

# Create and wrap the environment
env = make_vec_env(lambda: HumanoidEnv(render_mode='human'), n_envs=1)  # Wrap the environment properly

# Define the PPO model
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_humanoid_tensorboard/")
model = PPO.load("ppo_humanoid_model_muscles.zip")

# Train the model
model.learn(total_timesteps=10000)  # You can adjust the total_timesteps based on your requirements

# Save the trained model
model.save("ppo_humanoid_model_motor")

env.close()
# Load the trained model (if needed)
# model = PPO.load("ppo_humanoid_model")
