import os
import json
import uuid
import pandas as pd
from stable_baselines3 import PPO
from loguru import logger
from my_sol_model_gpt_try_one import DataCenterEnv
from given_info.utils import save_solution
from given_info.evaluation import evaluation_function

# Configure the loguru logger
logger.add("inference_log.log", format="{time} {level} {message}", level="INFO", rotation="1 MB", compression="zip")

def generate_solution_json():
    # Load the environment
    env = DataCenterEnv()

    # Load the trained model
    model_path = "./output/ppo_datacenter_model"
    if not os.path.exists(model_path + ".zip"):
        logger.error("Trained model not found at {}. Please ensure the model is trained and saved before running inference.", model_path)
        return

    model = PPO.load(model_path, env=env)

    # Reset the environment to start a new episode
    obs, _ = env.reset()  # Discard the info, only keep the observation
    done = False
    solution_actions = []

    while not done:
        # Predict the next action using the trained model
        action, _states = model.predict(obs, deterministic=True)

        # Apply the action to the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Record the actions
        for idx, act in enumerate(action):
            server_generation = env.servers['server_generation'].iloc[idx]
            datacenter_id = env.datacenters.sample(1)['datacenter_id'].iloc[0]

            if act == 0:  # Buy
                server_id = str(uuid.uuid4())  # Generate a new UUID for the new server
                solution_actions.append({
                    "time_step": env.current_time_step,
                    "datacenter_id": datacenter_id,
                    "server_generation": server_generation,
                    "server_id": server_id,
                    "action": "buy"
                })

            elif act == 1:  # Hold
                continue  # No action is recorded for holding

            elif act == 2:  # Move
                server_id = env.fleet[env.fleet['server_generation'] == server_generation]['server_id'].iloc[0]
                solution_actions.append({
                    "time_step": env.current_time_step,
                    "datacenter_id": datacenter_id,
                    "server_generation": server_generation,
                    "server_id": server_id,
                    "action": "move"
                })

            elif act == 3:  # Dismiss
                server_id = env.fleet[env.fleet['server_generation'] == server_generation]['server_id'].iloc[0]
                solution_actions.append({
                    "time_step": env.current_time_step,
                    "datacenter_id": datacenter_id,
                    "server_generation": server_generation,
                    "server_id": server_id,
                    "action": "dismiss"
                })

    # Ensure the output directory exists
    os.makedirs("./output", exist_ok=True)

    # Save the generated solution as a JSON file
    solution_path = "./output/generated_solution.json"
    with open(solution_path, 'w', encoding='utf-8') as f:
        json.dump(solution_actions, f, ensure_ascii=False, indent=4)

    logger.info("Generated solution saved to {}", solution_path)

    # Evaluate the generated solution
    solution_df = pd.DataFrame(solution_actions)
    final_score = evaluation_function(solution_df, env.demand, env.datacenters, env.servers, env.selling_prices, seed=123)
    if final_score is not None:
        logger.critical("Generated solution achieved a score of {}", final_score)
    else:
        logger.error("Failed to evaluate the generated solution.")

if __name__ == "__main__":
    generate_solution_json()
