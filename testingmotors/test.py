import gymnasium as gym
from stable_baselines3 import PPO
from humanoid_v4 import HumanoidEnv  # Ensure the correct path to the file

# Create the environment (ensure it matches the one used during training)
env = HumanoidEnv(render_mode='human')  # or use make_vec_env if you trained with vectorized env
env.reset()

# Load the pre-trained model
model = PPO.load("ppo_humanoid_model_motor.zip")  # Ensure this is the correct model path

# Run the pre-trained model in the environment
for _ in range(1000):  # Run for 1000 timesteps (or as needed)
    action, _states = model.predict(env.observation_space.sample())  # Random action
    env.step(action)  # Take a step in the environment

env.close()  # Close the environment once done
