from loguru import logger
from model import DataCenterEnv
from given_info.utils import load_problem_data
from stable_baselines3 import PPO

demand_df, datacenters_df, servers_df, selling_prices_df = load_problem_data()
# Initialize the environment
env = DataCenterEnv(demand_df, datacenters_df, servers_df, selling_prices_df)

# Initialize the model
model = PPO("MultiInputPolicy", env, verbose=1)

# Training loop
for i in range(1000000):  # Number of training iterations
    obs = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

    model.learn(total_timesteps=1, reset_num_timesteps=False)

    if i % 1000 == 0:  # Log every 1000 episodes
        logger.info(f"Episode {i}, Reward: {episode_reward}")

# Save the trained model
model.save("datacenter_model")