import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO

# 1. Register your custom environment (if necessary)
register(
    id='human-v0',  # The unique ID for your environment
    entry_point='humanoid_v4:HumanoidEnv',  # Replace with your actual environment
    max_episode_steps=1000,
)

# Load the environment (replace with your registered environment ID)
env = gym.make('human-v0', render_mode='human')  # Specify render_mode when making the environment

# Load the trained model
model = PPO.load("ppo_humanoid_model_muscles.zip")  # Replace with your model's path

# Reset the environment
obs, _ = env.reset()  # Unpack the observation and any info (second element, not used here)

# Testing loop
num_episodes = 5
for episode in range(num_episodes):
    done = False
    total_reward = 0
    while not done:
        # Predict the action based on the observation
        action, _states = model.predict(obs, deterministic=True)  # Get action using the model
        
        # Step in the environment with the predicted action
        obs, reward, trun, done, info = env.step(action)
         
        total_reward += reward
        
        # Visualize the environment
        env.render()  # Render without extra argument

    print(f"Episode {episode + 1} - Total Reward: {total_reward}")

env.close()
