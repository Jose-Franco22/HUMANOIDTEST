# import gymnasium as gym
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
# from humanoid_v4 import HumanoidEnv
# #from humanoidstandup_v4 import HumanoidStandupEnv #do later 


# gym.pprint_registry()

# # Create and wrap the environment
# env = make_vec_env(lambda: HumanoidEnv(render_mode='human'), n_envs=1)  # Wrap the environment properly

# # Define the PPO model
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_humanoid_tensorboard/")

# # Train the model
# model.learn(total_timesteps=5000000)  # You can adjust the total_timesteps based on your requirements

# # Save the trained model
# print("saving")

# model.save("ppo_humanoid_model_muscles")


# print("saved")

# env.close()
# Load the trained model (if needed)
# model = PPO.load("ppo_humanoid_model")






import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import os
from humanoid_v4 import HumanoidEnv  # Ensure you have the correct path to this file

# Registering or verifying that the environment exists (if necessary)
gym.pprint_registry()

# Create and wrap the environment
env = make_vec_env(lambda: HumanoidEnv(render_mode='human'), n_envs=1)  # Wrap the environment properly

# Ensure the environment is correctly initialized
assert env is not None, "Environment was not created properly!"

# Define a custom callback to save the model
class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq  # Frequency to save the model
        self.save_path = save_path  # Path where to save the model

    def _on_step(self) -> bool:
        # Check if the number of timesteps is a multiple of the save frequency
        if self.n_calls % self.save_freq == 0:
            self.model.save(os.path.join(self.save_path, f"ppo_humanoid_model_{self.n_calls}"))
            print(f"Model saved at timestep {self.n_calls}")
        return True

# Initialize PPO model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_humanoid_tensorboard/")  # Initialize model for training

# Alternatively, load the pre-trained model (if you're just testing the trained model)
# model = PPO.load("ppo_humanoid_model_motors_trained_v4.zip", env=env)

# Create the callback for saving the model every 100,000 timesteps
save_callback = SaveModelCallback(save_freq=2500000, save_path="./models")

# Train the model with the callback
model.learn(total_timesteps=30000000, callback=save_callback)  # You can adjust the total_timesteps based on your requirements

# Save the trained model at the end of training
model.save("ppo_humanoid_model_muscles_new_V1_test")

# Close the environment
env.close()

