# import os
# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# import pandas as pd
# import uuid
# from stable_baselines3 import PPO
# from loguru import logger
# from given_info.utils import load_problem_data, load_solution, save_solution
# from given_info.evaluation import evaluation_function, get_actual_demand, check_datacenter_slots_size_constraint, update_check_lifespan
#
# # Configure the loguru logger
# logger.add("data_center_env.log", format="{time} {level} {message}", level="INFO", rotation="1 MB", compression="zip")
#
# class DataCenterEnv(gym.Env):
#     def __init__(self):
#         super(DataCenterEnv, self).__init__()
#
#         # Load data from CSV files
#         self.demand, self.datacenters, self.servers, self.selling_prices = load_problem_data(r"../given_info/data")
#
#         # Initialize environment parameters
#         self.current_time_step = 1
#         self.total_time_steps = 168
#         self.baseline_score = 409082658.25172067  # Baseline score for comparison
#         self.fleet = pd.DataFrame(columns=self.servers.columns)  # Initialize fleet with correct columns
#
#         # Action space: [Buy, Hold, Move, Dismiss] for each server generation (MultiDiscrete for different actions per server)
#         self.action_space = spaces.MultiDiscrete([4] * len(self.servers['server_generation'].unique()))
#
#         # Observation space: Flattened state vector combining relevant features from all data centers and servers
#         self.observation_space = spaces.Box(low=0, high=1, shape=(self._calculate_observation_space_shape(),), dtype=np.float32)
#
#         # Initialize state
#         self.state = self._initialize_state()
#
#         # Initial reward tracking
#         self.total_reward = 0
#         self.last_score = 0
#
#         # Ensure output directory exists
#         os.makedirs("./output", exist_ok=True)
#
#         logger.info("DataCenterEnv initialized with baseline score of {}", self.baseline_score)
#
#     def _calculate_observation_space_shape(self):
#         return len(self.datacenters.columns) + len(self.servers.columns) * len(self.datacenters)
#
#     def _initialize_state(self):
#         state = np.zeros(self._calculate_observation_space_shape())
#         return state
#
#     def reset(self, **kwargs):
#         self.current_time_step = 1
#         self.state = self._initialize_state()
#         self.fleet = pd.DataFrame(columns=self.servers.columns)  # Reset fleet with correct columns
#         self.total_reward = 0
#         self.last_score = 0
#         logger.info("Environment reset")
#         return self.state, {}
#
#     def step(self, action):
#         logger.info("Applied action {}", action)
#         reward, valid = self._apply_action(action)
#         self.state = self._update_state()
#
#         if self.current_time_step >= self.total_time_steps:
#             solution = self._generate_solution_json()
#             final_score = evaluation_function(solution, self.demand, self.datacenters, self.servers,
#                                               self.selling_prices, seed=123)
#
#             if final_score is not None:
#                 if final_score > self.baseline_score:
#                     reward += 1000  # Super big reward for exceeding baseline score
#                 else:
#                     reward += 100  # Big reward for finishing the episode
#                 self.last_score = final_score
#                 logger.info("Final score: {}", final_score)
#             else:
#                 reward -= 500  # Large penalty if evaluation fails
#                 logger.error("Evaluation failed, final score is None")
#
#             terminated = True  # Episode ends
#             truncated = False  # Not truncated by a time limit
#
#             if self._is_solution_valid(solution):
#                 save_solution(solution, f'./output/valid_solution_{self.last_score}.json')
#                 logger.info("Valid solution saved with score {}", self.last_score)
#         else:
#             terminated = False
#             truncated = False
#             reward += 10 if valid else -50  # Small reward for correct actions, big penalty for incorrect
#
#         self.total_reward += reward
#         self.current_time_step += 1
#         info = {}
#         return self.state, reward, terminated, truncated, info
#
#     def _apply_action(self, action):
#         valid_action = True
#         for idx, act in enumerate(action):
#             server_generation = self.servers['server_generation'].iloc[idx]
#             if act == 0:  # Buy
#                 valid_action = self._buy_server(server_generation)
#             elif act == 1:  # Hold
#                 continue  # Do nothing, hold servers
#             elif act == 2:  # Move
#                 valid_action = self._move_server(server_generation)
#             elif act == 3:  # Dismiss
#                 valid_action = self._dismiss_server(server_generation)
#             if not valid_action:
#                 break
#
#         logger.info("Action {} applied with validity {}", action, valid_action)
#         return (10 if valid_action else -50), valid_action
#
#     def _buy_server(self, server_generation):
#         available_servers = self.servers[self.servers['server_generation'] == server_generation]
#         if not available_servers.empty:
#             new_server = available_servers.sample(1).copy()
#             new_server['server_id'] = str(uuid.uuid4())  # Assign unique UUID
#             new_server['datacenter_id'] = self.datacenters.sample(1)['datacenter_id'].iloc[0]
#             new_server['lifespan'] = 0  # Initialize lifespan
#             self.fleet = pd.concat([self.fleet, new_server], ignore_index=True)
#             logger.info("Bought server: {} in datacenter: {}", new_server['server_generation'].values[0], new_server['datacenter_id'].values[0])
#             return True
#         logger.warning("No available servers for generation: {}", server_generation)
#         return False
#
#     def _move_server(self, server_generation):
#         if server_generation in self.fleet['server_generation'].values:
#             moving_servers = self.fleet[self.fleet['server_generation'] == server_generation]
#             if not moving_servers.empty:
#                 server_to_move = moving_servers.sample(1)
#                 new_datacenter_id = self.datacenters.sample(1)['datacenter_id'].iloc[0]
#                 self.fleet.loc[server_to_move.index, 'datacenter_id'] = new_datacenter_id
#                 self.fleet.loc[server_to_move.index, 'moved'] = 1
#                 logger.info("Moved server {} to datacenter {}", server_generation, new_datacenter_id)
#                 return True
#         logger.warning("No servers to move for generation: {}", server_generation)
#         return False
#
#     def _dismiss_server(self, server_generation):
#         if server_generation in self.fleet['server_generation'].values:
#             dismissing_servers = self.fleet[self.fleet['server_generation'] == server_generation]
#             if not dismissing_servers.empty:
#                 self.fleet = self.fleet.drop(dismissing_servers.sample(1).index)
#                 logger.info("Dismissed server {}", server_generation)
#                 return True
#         logger.warning("No servers to dismiss for generation: {}", server_generation)
#         return False
#
#     def _update_state(self):
#         if 'lifespan' in self.fleet.columns:
#             self.fleet['lifespan'] += 1  # Increment lifespan of each server
#         state = np.zeros(self._calculate_observation_space_shape())
#         return state
#
#     def _generate_solution_json(self):
#         solution = self.fleet.copy()
#         solution['time_step'] = self.current_time_step
#         solution['action'] = 'hold'
#         solution = solution.merge(self.datacenters, on='datacenter_id', how='left')  # Merge to ensure slots_capacity is included
#         return solution
#
#     def _is_solution_valid(self, solution):
#         try:
#             check_datacenter_slots_size_constraint(solution)
#             solution = update_check_lifespan(solution)
#             return True
#         except Exception as e:
#             logger.error("Invalid solution: {}", e)
#             return False
#
# # Usage Example
# env = DataCenterEnv()
# model = PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=100000)
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import uuid
from stable_baselines3 import PPO
from loguru import logger
from given_info.utils import load_problem_data, load_solution, save_solution
from given_info.evaluation import evaluation_function, get_actual_demand, check_datacenter_slots_size_constraint, update_check_lifespan

# Configure the loguru logger
# logger.add("data_center_env.log", format="{time} {level} {message}", level="INFO", rotation="1 MB", compression="zip")

class DataCenterEnv(gym.Env):
    @logger.catch
    def __init__(self):
        super(DataCenterEnv, self).__init__()

        # Load data from CSV files
        self.demand, self.datacenters, self.servers, self.selling_prices = load_problem_data(r"../given_info/data")

        # Initialize environment parameters
        self.current_time_step = 1
        self.total_time_steps = 168
        self.baseline_score = 409082658.25172067  # Baseline score for comparison
        self.fleet = pd.DataFrame(columns=self.servers.columns)  # Initialize fleet with correct columns

        # Action space: [Buy, Hold, Move, Dismiss] for each server generation (MultiDiscrete for different actions per server)
        self.action_space = spaces.MultiDiscrete([4] * len(self.servers['server_generation'].unique()))

        # Observation space: Flattened state vector combining relevant features from all data centers and servers
        self.observation_space = spaces.Box(low=0, high=1, shape=(self._calculate_observation_space_shape(),), dtype=np.float32)

        # Initialize state
        self.state = self._initialize_state()

        # Initial reward tracking
        self.total_reward = 0
        self.last_score = 0

        # Ensure output directory exists
        os.makedirs("./output", exist_ok=True)

        logger.info("DataCenterEnv initialized with baseline score of {}", self.baseline_score)

    @logger.catch
    def _calculate_observation_space_shape(self):
        return len(self.datacenters.columns) + len(self.servers.columns) * len(self.datacenters)

    @logger.catch
    def _initialize_state(self):
        state = np.zeros(self._calculate_observation_space_shape())
        return state

    @logger.catch
    def reset(self, **kwargs):
        self.current_time_step = 1
        self.state = self._initialize_state()
        self.fleet = pd.DataFrame(columns=self.servers.columns)  # Reset fleet with correct columns
        self.total_reward = 0
        self.last_score = 0
        logger.info("Environment reset")
        return self.state, {}

    @logger.catch
    def step(self, action):
        logger.info("Applied action {}", action)
        reward, valid = self._apply_action(action)
        self.state = self._update_state()

        if self.current_time_step >= self.total_time_steps:
            solution = self._generate_solution_json()
            final_score = evaluation_function(solution, self.demand, self.datacenters, self.servers,
                                              self.selling_prices, seed=123)

            if final_score is not None:
                if final_score > self.baseline_score:
                    reward += 1000  # Super big reward for exceeding baseline score
                else:
                    reward += 100  # Big reward for finishing the episode
                self.last_score = final_score
                logger.info("Final score: {}", final_score)
                logger.critical("Training cycle complete. Achieved score: {}", final_score)
            else:
                reward -= 500  # Large penalty if evaluation fails
                logger.error("Evaluation failed, final score is None")

            terminated = True  # Episode ends
            truncated = False  # Not truncated by a time limit

            if self._is_solution_valid(solution):
                save_solution(solution, f'./output/valid_solution_{self.last_score}.json')
                logger.info("Valid solution saved with score {}", self.last_score)
        else:
            terminated = False
            truncated = False
            reward += 10 if valid else -50  # Small reward for correct actions, big penalty for incorrect

        self.total_reward += reward
        self.current_time_step += 1
        info = {}
        return self.state, reward, terminated, truncated, info

    @logger.catch
    def _apply_action(self, action):
        valid_action = True
        for idx, act in enumerate(action):
            server_generation = self.servers['server_generation'].iloc[idx]
            if act == 0:  # Buy
                valid_action = self._buy_server(server_generation)
            elif act == 1:  # Hold
                continue  # Do nothing, hold servers
            elif act == 2:  # Move
                valid_action = self._move_server(server_generation)
            elif act == 3:  # Dismiss
                valid_action = self._dismiss_server(server_generation)
            if not valid_action:
                break

        logger.info("Action {} applied with validity {}", action, valid_action)
        return (10 if valid_action else -50), valid_action

    @logger.catch
    def _buy_server(self, server_generation):
        available_servers = self.servers[self.servers['server_generation'] == server_generation]
        if not available_servers.empty:
            new_server = available_servers.sample(1).copy()
            new_server['server_id'] = str(uuid.uuid4())  # Assign unique UUID
            new_server['datacenter_id'] = self.datacenters.sample(1)['datacenter_id'].iloc[0]
            new_server['lifespan'] = 0  # Initialize lifespan
            self.fleet = pd.concat([self.fleet, new_server], ignore_index=True)
            logger.info("Bought server: {} in datacenter: {}", new_server['server_generation'].values[0], new_server['datacenter_id'].values[0])
            return True
        logger.warning("No available servers for generation: {}", server_generation)
        return False

    @logger.catch
    def _move_server(self, server_generation):
        if server_generation in self.fleet['server_generation'].values:
            moving_servers = self.fleet[self.fleet['server_generation'] == server_generation]
            if not moving_servers.empty:
                server_to_move = moving_servers.sample(1)
                new_datacenter_id = self.datacenters.sample(1)['datacenter_id'].iloc[0]
                self.fleet.loc[server_to_move.index, 'datacenter_id'] = new_datacenter_id
                self.fleet.loc[server_to_move.index, 'moved'] = 1
                logger.info("Moved server {} to datacenter {}", server_generation, new_datacenter_id)
                return True
        logger.warning("No servers to move for generation: {}", server_generation)
        return False

    @logger.catch
    def _dismiss_server(self, server_generation):
        if server_generation in self.fleet['server_generation'].values:
            dismissing_servers = self.fleet[self.fleet['server_generation'] == server_generation]
            if not dismissing_servers.empty:
                self.fleet = self.fleet.drop(dismissing_servers.sample(1).index)
                logger.info("Dismissed server {}", server_generation)
                return True
        logger.warning("No servers to dismiss for generation: {}", server_generation)
        return False

    @logger.catch
    def _update_state(self):
        if 'lifespan' in self.fleet.columns:
            self.fleet['lifespan'] += 1  # Increment lifespan of each server
        state = np.zeros(self._calculate_observation_space_shape())
        return state

    @logger.catch
    def _generate_solution_json(self):
        solution = self.fleet.copy()
        solution['time_step'] = self.current_time_step
        solution['action'] = 'hold'
        solution = solution.merge(self.datacenters, on='datacenter_id', how='left')  # Merge to ensure slots_capacity is included
        return solution

    @logger.catch
    def _is_solution_valid(self, solution):
        try:
            check_datacenter_slots_size_constraint(solution)
            solution = update_check_lifespan(solution)
            return True
        except Exception as e:
            logger.error("Invalid solution: {}", e)
            return False

# Usage Example

# Configure the loguru logger
# logger.add("training_log.log", format="{time} {level} {message}", level="WARNING", rotation="1 MB", compression="zip")

@logger.catch
# Train the model
def train_model():
    env = DataCenterEnv()

    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=168)

    # Save the model
    model_path = "./output/ppo_datacenter_model"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    logger.critical("Model trained and saved successfully at {}", model_path)


if __name__ == "__main__":
    train_model()
