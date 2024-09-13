# import gymnasium
# from gymnasium import spaces
# import numpy as np
# import pandas as pd
# import uuid
# import json
# from scipy.sparse import lil_matrix
# from given_info.utils import load_problem_data
# from given_info.evaluation import evaluation_function
# from loguru import logger
#
#
# class DataCenterEnv(gymnasium.Env):
#     def __init__(self, demand_df, datacenters_df, servers_df, selling_prices_df):
#         super(DataCenterEnv, self).__init__()
#
#         self.demand_df = demand_df
#         self.datacenters_df = datacenters_df
#         self.servers_df = servers_df
#         self.selling_prices_df = selling_prices_df
#
#         self.previous_score = 0
#         self.current_score = None
#
#         self.dc_id_to_idx = {dc_id: idx for idx, dc_id in enumerate(datacenters_df['datacenter_id'])}
#         self.idx_to_dc_id = {v: k for k, v in self.dc_id_to_idx.items()}
#
#         unique_server_gens = self.servers_df['server_generation'].dropna().unique()
#         self.server_type_map = {i: server_gen for i, server_gen in enumerate(unique_server_gens)}
#         logger.debug(f"Server type map: {self.server_type_map}")
#
#         self.n_datacenters = len(self.datacenters_df)
#         self.n_server_types = len(self.server_type_map)
#
#         self.action_space = spaces.Dict({
#             'buy': spaces.MultiDiscrete([self.n_server_types, self.n_datacenters, 101]),
#             'move': spaces.MultiDiscrete([10000, self.n_datacenters])
#         })
#
#         self.observation_space = spaces.Dict({
#             'demand': spaces.Box(low=0, high=np.inf, shape=(self.n_server_types,), dtype=np.int32),
#             'datacenters': spaces.Box(low=0, high=np.inf, shape=(self.n_datacenters * 3,), dtype=np.float32),
#             'servers': spaces.Box(low=0, high=np.inf, shape=(self.n_datacenters * self.n_server_types,),
#                                   dtype=np.float32)
#         })
#
#         self.state = None
#         self.current_time_step = 0
#         self.servers = {}
#         self.action_log = []
#
#         self.valid_moves_cache = {}
#         self.datacenter_capacities = {dc: capacity for dc, capacity in
#                                       zip(self.datacenters_df['datacenter_id'], self.datacenters_df['slots_capacity'])}
#
#     def reset(self):
#         self.current_time_step = 1
#         self.servers = {}
#         self.action_log = []
#         self.state = self._get_initial_state()
#         self.current_score = None
#         self.valid_moves_cache = {}
#         self.datacenter_capacities = {dc: capacity for dc, capacity in
#                                       zip(self.datacenters_df['datacenter_id'], self.datacenters_df['slots_capacity'])}
#
#         logger.info(f"Environment reset at time_step {self.current_time_step}")
#         return self.state
#
#     def step(self, actions):
#         logger.debug(f"Actions received: {actions}")
#         reward = 0
#
#         self._decrease_lifespan()
#
#         for action in actions['buy']:
#             server_type, dc_idx, num_to_buy = action[:3]
#             datacenter_id = self.idx_to_dc_id.get(dc_idx)
#             server_gen = self.server_type_map.get(server_type)
#             if datacenter_id is not None and server_gen is not None:
#                 logger.debug(
#                     f"Processing buy action: server_type={server_gen}, datacenter_id={datacenter_id}, num_to_buy={num_to_buy}")
#                 if self._can_purchase(server_type, datacenter_id, num_to_buy):
#                     self._buy_server(server_gen, datacenter_id, num_to_buy)
#
#         moved_servers = set()
#         for action in actions['move']:
#             server_uuid_index, target_dc_idx = action[:2]
#             server_uuid = self._get_uuid_by_index(server_uuid_index)
#             target_datacenter_id = self.idx_to_dc_id.get(target_dc_idx)
#             if server_uuid and target_datacenter_id is not None:
#                 logger.debug(
#                     f"Processing move action: server_uuid={server_uuid}, target_datacenter_id={target_datacenter_id}")
#                 if self._can_move(server_uuid, target_datacenter_id):
#                     self._move_server(server_uuid, target_datacenter_id)
#                     moved_servers.add(server_uuid)
#
#         self._auto_hold_and_dismiss(moved_servers)
#
#         done = self._check_done()
#
#         if done:
#             solution = self.get_action_log()
#             logger.info(f"Evaluating solution at time_step {self.current_time_step}")
#
#             self.current_score = evaluation_function(solution, self.demand_df, self.datacenters_df, self.servers_df,
#                                                      self.selling_prices_df, seed=123)
#             logger.info(f"Current score: {self.current_score}, Previous score: {self.previous_score}")
#
#             if self.current_score is not None:
#                 reward = self.current_score - self.previous_score
#             else:
#                 reward = 0
#                 logger.warning("Invalid actions taken, current_score is None")
#
#             self.previous_score = self.current_score if self.current_score is not None else self.previous_score
#             self.write_action_log_to_json("solution_inside_model_nick.json")
#             self.reset()
#
#         self.current_time_step += 1
#
#         next_state = self._get_next_state() if not done else self.state
#         return next_state, reward, done, {}
#
#     def _decrease_lifespan(self):
#         for uuid, server_data in self.servers.items():
#             server_data['remaining_lifespan'] -= 1
#         logger.debug(f"Lifespan decreased for all servers. Current server states: {self.servers}")
#
#     def _get_initial_state(self):
#         demand = self.demand_df.iloc[self.current_time_step - 1, 2:].values
#         datacenters = self.datacenters_df[['slots_capacity', 'cost_of_energy', 'latency_sensitivity']].values.flatten()
#         servers = np.zeros((self.n_datacenters * self.n_server_types,))
#         logger.debug(f"Initial state set with demand: {demand}, datacenters: {datacenters}, servers: {servers}")
#         return {'demand': demand, 'datacenters': datacenters, 'servers': servers}
#
#     def _get_next_state(self):
#         demand = self.demand_df.iloc[self.current_time_step, 2:].values
#         datacenters = self.datacenters_df[['slots_capacity', 'cost_of_energy', 'latency_sensitivity']].values.flatten()
#         servers = np.zeros((self.n_datacenters * self.n_server_types,))
#         logger.debug(f"Next state prepared with demand: {demand}, datacenters: {datacenters}, servers: {servers}")
#         return {'demand': demand, 'datacenters': datacenters, 'servers': servers}
#
#     def _buy_server(self, server_gen, datacenter_id, num_to_buy):
#         server_info = self.servers_df[self.servers_df['server_generation'] == server_gen]
#         if server_info.empty:
#             logger.error(f"No matching server generation found for server_type: {server_gen}")
#             return
#
#         server_info = server_info.iloc[0]
#         slots_required = server_info['slots_size']
#         lifespan = server_info['life_expectancy']
#
#         for _ in range(num_to_buy):
#             new_uuid = str(uuid.uuid4())
#             self.servers[new_uuid] = {
#                 'datacenter_id': datacenter_id,
#                 'server_type': server_gen,
#                 'remaining_lifespan': lifespan,
#                 'slots': slots_required
#             }
#             self._log_action("buy", new_uuid, datacenter_id, server_gen)
#             self._update_valid_moves_cache(new_uuid)
#
#             self.datacenters_df.loc[
#                 self.datacenters_df['datacenter_id'] == datacenter_id, 'slots_capacity'] -= slots_required
#             self.datacenter_capacities[datacenter_id] -= slots_required
#
#         self.demand_df.loc[self.current_time_step - 1, server_gen] -= num_to_buy
#         logger.debug(f"Bought {num_to_buy} servers of type {server_gen} for datacenter {datacenter_id}")
#
#     def _move_server(self, server_uuid, target_dc):
#         if server_uuid in self.servers:
#             current_dc = self.servers[server_uuid]['datacenter_id']
#             slots = self.servers[server_uuid]['slots']
#             self.servers[server_uuid]['datacenter_id'] = target_dc
#
#             server_generation = self.servers[server_uuid]['server_type']
#             self._log_action("move", server_uuid, target_dc, server_generation)
#
#             self.datacenters_df.loc[self.datacenters_df['datacenter_id'] == current_dc, 'slots_capacity'] += slots
#             self.datacenters_df.loc[self.datacenters_df['datacenter_id'] == target_dc, 'slots_capacity'] -= slots
#             self.datacenter_capacities[current_dc] += slots
#             self.datacenter_capacities[target_dc] -= slots
#
#             self._update_valid_moves_cache(server_uuid)
#
#             logger.debug(f"Moved server {server_uuid} from datacenter {current_dc} to {target_dc}")
#
#     def _auto_hold_and_dismiss(self, moved_servers):
#         to_dismiss = []
#
#         for uuid, server_data in self.servers.items():
#             if uuid not in moved_servers:
#                 if server_data['remaining_lifespan'] <= 0:
#                     to_dismiss.append(uuid)
#                 else:
#                     self._log_action("hold", uuid, server_data['datacenter_id'], server_data['server_type'])
#
#         for uuid in to_dismiss:
#             self._dismiss_server(uuid)
#         logger.debug(f"Auto-hold and dismiss processed. Servers to dismiss: {to_dismiss}")
#
#     def _dismiss_server(self, uuid):
#         if uuid in self.servers:
#             server_data = self.servers[uuid]
#             datacenter_id = server_data['datacenter_id']
#             slots_freed = server_data['slots']
#             server_generation = server_data['server_type']
#
#             self._log_action("dismiss", uuid, datacenter_id, server_generation)
#
#             self.datacenters_df.loc[
#                 self.datacenters_df['datacenter_id'] == datacenter_id, 'slots_capacity'] += slots_freed
#             self.datacenter_capacities[datacenter_id] += slots_freed
#
#             del self.servers[uuid]
#             if uuid in self.valid_moves_cache:
#                 del self.valid_moves_cache[uuid]
#             logger.debug(f"Dismissed server {uuid} from datacenter {datacenter_id}. Freed {slots_freed} slots.")
#
#     def _log_action(self, action, server_id, datacenter_id, server_generation):
#         action_entry = {
#             "time_step": self.current_time_step,
#             "datacenter_id": datacenter_id,
#             "server_generation": server_generation,
#             "server_id": server_id,
#             "action": action
#         }
#         self.action_log.append(action_entry)
#         logger.debug(f"Action logged: {action_entry}")
#
#     def _check_done(self):
#         return self.current_time_step >= 168
#
#     def _can_purchase(self, server_type, datacenter_id, num_to_buy):
#         server_gen = self.server_type_map[server_type]
#         server_info = self.servers_df[self.servers_df['server_generation'] == server_gen]
#
#         if server_info.empty:
#             logger.error(f"No matching server generation found for server_type: {server_gen}")
#             return False
#
#         server_info = server_info.iloc[0]
#         slots_required = server_info['slots_size'] * num_to_buy
#
#         dc_capacity = self.datacenter_capacities[datacenter_id]
#         demand_available = self.demand_df.loc[self.current_time_step - 1, server_gen]
#         can_purchase = dc_capacity >= slots_required and num_to_buy <= demand_available
#         logger.debug(
#             f"Can purchase check for server_type={server_type} (server_generation={server_gen}), datacenter_id={datacenter_id}, num_to_buy={num_to_buy}: {can_purchase}")
#         return can_purchase
#
#     def _can_move(self, server_uuid, target_dc):
#         if server_uuid not in self.servers:
#             logger.error(f"Server UUID {server_uuid} not found for moving.")
#             return False
#         current_dc = self.servers[server_uuid]['datacenter_id']
#         slots_required = self.servers[server_uuid]['slots']
#         target_dc_capacity = self.datacenter_capacities[target_dc]
#         can_move = target_dc_capacity >= slots_required and current_dc != target_dc
#         logger.debug(f"Can move check for server_uuid={server_uuid}, target_dc={target_dc}: {can_move}")
#         return can_move
#
#     def _get_uuid_by_index(self, server_index):
#         if server_index < len(self.servers):
#             uuid = list(self.servers.keys())[server_index]
#             logger.debug(f"UUID found for index {server_index}: {uuid}")
#             return uuid
#         logger.error(f"Server index {server_index} is out of range.")
#         return None
#
#     def _update_valid_moves_cache(self, server_uuid=None):
#         if server_uuid is None:
#             self.valid_moves_cache = {}
#             for uuid, server_data in self.servers.items():
#                 self._update_server_valid_moves(uuid, server_data)
#         else:
#             server_data = self.servers[server_uuid]
#             self._update_server_valid_moves(server_uuid, server_data)
#
#     def _update_server_valid_moves(self, uuid, server_data):
#         current_dc = server_data['datacenter_id']
#         slots = server_data['slots']
#         self.valid_moves_cache[uuid] = [
#             dc for dc, capacity in self.datacenter_capacities.items()
#             if dc != current_dc and capacity >= slots
#         ]
#
#     def _get_action_mask(self):
#         buy_mask = self._get_buy_mask()
#         move_mask = self._get_move_mask()
#         return {'buy': buy_mask, 'move': move_mask}
#
#     def _get_buy_mask(self):
#         buy_mask = np.zeros((self.n_server_types, self.n_datacenters, 101), dtype=np.int8)
#
#         demand = self.demand_df.iloc[self.current_time_step - 1, 2:].values
#         dc_capacities = np.array([self.datacenter_capacities[dc] for dc in self.datacenters_df['datacenter_id']])
#
#         logger.debug(f"Current demand: {demand}")
#         logger.debug(f"Current datacenter capacities: {dc_capacities}")
#
#         for server_type in range(self.n_server_types):
#             server_gen = self.server_type_map[server_type]
#             if pd.isna(server_gen):
#                 continue
#             server_info = self.servers_df[self.servers_df['server_generation'] == server_gen].iloc[0]
#             slots_per_server = server_info['slots_size']
#
#             max_per_dc = np.minimum(demand[server_type], dc_capacities // slots_per_server)
#             max_per_dc = np.minimum(max_per_dc, 100).astype(int)
#
#             logger.debug(f"Server type {server_gen}: max_per_dc = {max_per_dc}")
#
#             for dc_id in range(self.n_datacenters):
#                 buy_mask[server_type, dc_id, :max_per_dc[dc_id] + 1] = 1
#
#         logger.debug(f"Buy mask sum: {buy_mask.sum()}")
#         return buy_mask
#
#     def _get_move_mask(self):
#         move_mask = lil_matrix((len(self.servers), self.n_datacenters), dtype=np.int8)
#         for i, (uuid, valid_dcs) in enumerate(self.valid_moves_cache.items()):
#             for dc in valid_dcs:
#                 move_mask[i, self.dc_id_to_idx[dc]] = 1
#
#         logger.debug(f"Move mask sum: {move_mask.sum()}")
#         return move_mask
#
#     def get_action_log(self):
#         return self.action_log
#
#     def write_action_log_to_json(self, filename):
#         with open(filename, 'w') as f:
#             json.dump(self.action_log, f, indent=2)
#         logger.info(f"Action log with {len(self.action_log)} entries written to {filename}")
#
# # Main execution
#
# if __name__ == "__main__":
#     # Load data
#     demand_df, datacenters_df, servers_df, selling_prices_df = load_problem_data(r"../given_info/data")
#
#     # Create environment
#     env = DataCenterEnv(demand_df, datacenters_df, servers_df, selling_prices_df)
#
#     # Run simulation
#     obs = env.reset()
#     done = False
#     total_reward = 0
#
#     for time_step in range(1, 169):  # Ensure we go through all 168 time steps
#         logger.info(f"Processing time step {time_step}")
#         action_mask = env._get_action_mask()
#
#         buy_actions = []
#         move_actions = []
#
#         # Generate buy actions
#         valid_buys = np.where(action_mask['buy'])
#         logger.debug(f"Valid buy actions: {valid_buys}")
#         if len(valid_buys[0]) > 0:
#             buy_indices = np.random.choice(len(valid_buys[0]), size=min(5, len(valid_buys[0])), replace=False)
#             for idx in buy_indices:
#                 server_type, dc_id, num_to_buy = valid_buys[0][idx], valid_buys[1][idx], valid_buys[2][idx]
#                 buy_actions.append((server_type, dc_id, num_to_buy))
#
#         # Generate move actions
#         valid_moves = action_mask['move'].nonzero()
#         logger.debug(f"Valid move actions: {valid_moves}")
#         if len(valid_moves[0]) > 0:
#             move_indices = np.random.choice(len(valid_moves[0]), size=min(5, len(valid_moves[0])), replace=False)
#             for idx in move_indices:
#                 server_index, target_dc = valid_moves[0][idx], valid_moves[1][idx]
#                 move_actions.append((server_index, target_dc))
#
#         logger.info(f"Generated actions - Buy: {buy_actions}, Move: {move_actions}")
#
#         # Take step in environment
#         obs, reward, done, _ = env.step({'buy': buy_actions, 'move': move_actions})
#         total_reward += reward
#
#         logger.info(f"Time step: {env.current_time_step}, Reward: {reward}, Total Reward: {total_reward}")
#         logger.info(f"Current action log length: {len(env.action_log)}")
#
#         if done:
#             break
#
#     logger.info(f"Simulation completed. Final score: {env.current_score}")
#     logger.info(f"Total actions logged: {len(env.action_log)}")
#
#     # Write action log to JSON file
#     json_filename = 'solution.json'
#     env.write_action_log_to_json(json_filename)
#     logger.info(f"Action log written to {json_filename}")
#
#     # Evaluate the solution
#     score = evaluation_function(json_filename, demand_df, datacenters_df, servers_df, selling_prices_df, seed=123)
#
#     if score:
#         logger.info(f"Positive score achieved: {score}. Saving solution.")
#         env.write_action_log_to_json(f'solution_score_{score:.2f}.json')
#     else:
#         logger.info(f"Score not positive: {score}. Solution not saved.")
#         env.write_action_log_to_json('solution_score_failing.json')
#
#     logger.info(f"Final action log length: {len(env.action_log)}")

import gymnasium
import torch
from gymnasium import spaces
import numpy as np
import pandas as pd
import uuid
import json
from scipy.sparse import lil_matrix
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from given_info.utils import load_problem_data, load_solution
from given_info.evaluation import evaluation_function
from loguru import logger


class DataCenterEnv(gymnasium.Env):
    def __init__(self, demand_df, datacenters_df, servers_df, selling_prices_df):
        super(DataCenterEnv, self).__init__()

        self.demand_df = demand_df
        self.datacenters_df = datacenters_df
        self.servers_df = servers_df
        self.selling_prices_df = selling_prices_df

        self.previous_score = 0
        self.current_score = None

        self.dc_id_to_idx = {dc_id: idx for idx, dc_id in enumerate(datacenters_df['datacenter_id'])}
        self.idx_to_dc_id = {v: k for k, v in self.dc_id_to_idx.items()}

        unique_server_gens = self.servers_df['server_generation'].dropna().unique()
        self.server_type_map = {i: server_gen for i, server_gen in enumerate(unique_server_gens)}
        logger.debug(f"Server type map: {self.server_type_map}")

        self.n_datacenters = len(self.datacenters_df)
        self.n_server_types = len(self.server_type_map)

        self.action_space = spaces.MultiDiscrete([
            self.n_server_types,  # server type for buy action
            self.n_datacenters,  # datacenter for buy action
            101,  # number to buy
            10000,  # server index for move action
            self.n_datacenters  # target datacenter for move action
        ])

        self.observation_space = spaces.Dict({
            'demand': spaces.Box(low=0, high=np.inf, shape=(self.n_server_types,), dtype=np.float32),
            'datacenters': spaces.Box(low=0, high=np.inf, shape=(self.n_datacenters * 3,), dtype=np.float32),
            'servers': spaces.Box(low=0, high=np.inf, shape=(self.n_datacenters * self.n_server_types,),
                                  dtype=np.float32)
        })

        self.state = None
        self.current_time_step = 0
        self.servers = {}
        self.action_log = []

        self.valid_moves_cache = {}
        self.datacenter_capacities = {dc: capacity for dc, capacity in
                                      zip(self.datacenters_df['datacenter_id'], self.datacenters_df['slots_capacity'])}
        self.tmp_actions_were_taken = set()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # This line is important for seeding the environment

        self.current_time_step = 1
        self.servers = {}
        self.action_log = []
        self.state = self._get_initial_state()
        self.current_score = None
        self.valid_moves_cache = {}
        self.datacenter_capacities = {dc: capacity for dc, capacity in
                                      zip(self.datacenters_df['datacenter_id'], self.datacenters_df['slots_capacity'])}
        self.tmp_actions_were_taken = set()

        logger.info(f"Environment reset at time_step {self.current_time_step}")
        return self.state, {}  # Return the state and an empty info dict

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_time_step = 1
        self.servers = {}
        self.action_log = []
        self.state = self._get_initial_state()
        self.current_score = None
        self.valid_moves_cache = {}
        self.datacenter_capacities = {dc: capacity for dc, capacity in
                                      zip(self.datacenters_df['datacenter_id'], self.datacenters_df['slots_capacity'])}
        self.tmp_actions_were_taken = set()

        # Update action space
        self.action_space = spaces.MultiDiscrete([
            self.n_server_types,  # server type for buy action
            self.n_datacenters,  # datacenter for buy action
            101,  # number to buy
            max(1, self.get_num_servers()),  # server index for move action (at least 1)
            self.n_datacenters  # target datacenter for move action
        ])

        logger.info(f"Environment reset at time_step {self.current_time_step}")
        return self.state, {}
    def _get_initial_state(self):
        demand = self.demand_df.iloc[self.current_time_step - 1, 2:].values.astype(np.float32)

        datacenters = self.datacenters_df[['slots_capacity', 'cost_of_energy']].values.astype(np.float32)
        latency_sensitivity = np.array(
            [self._encode_latency_sensitivity(v) for v in self.datacenters_df['latency_sensitivity']], dtype=np.float32)
        datacenters = np.column_stack((datacenters, latency_sensitivity)).flatten()

        servers = np.zeros((self.n_datacenters * self.n_server_types,), dtype=np.float32)

        logger.debug(f"Initial state set with demand: {demand}, datacenters: {datacenters}, servers: {servers}")
        return {'demand': demand, 'datacenters': datacenters, 'servers': servers}

    def _get_next_state(self):
        demand = self.demand_df.iloc[self.current_time_step, 2:].values.astype(np.float32)

        datacenters = self.datacenters_df[['slots_capacity', 'cost_of_energy']].values.astype(np.float32)
        latency_sensitivity = np.array(
            [self._encode_latency_sensitivity(v) for v in self.datacenters_df['latency_sensitivity']], dtype=np.float32)
        datacenters = np.column_stack((datacenters, latency_sensitivity)).flatten()

        servers = np.zeros((self.n_datacenters * self.n_server_types,), dtype=np.float32)

        logger.debug(f"Next state prepared with demand: {demand}, datacenters: {datacenters}, servers: {servers}")
        return {'demand': demand, 'datacenters': datacenters, 'servers': servers}
    def _buy_server(self, server_gen, datacenter_id, num_to_buy):
        server_info = self.servers_df[self.servers_df['server_generation'] == server_gen]
        if server_info.empty:
            logger.error(f"No matching server generation found for server_type: {server_gen}")
            return

        server_info = server_info.iloc[0]
        slots_required = server_info['slots_size']
        lifespan = server_info['life_expectancy']

        for _ in range(num_to_buy):
            new_uuid = str(uuid.uuid4())
            self.servers[new_uuid] = {
                'datacenter_id': datacenter_id,
                'server_type': server_gen,
                'remaining_lifespan': lifespan,
                'slots': slots_required
            }
            if new_uuid not in self.tmp_actions_were_taken:
                self._log_action("buy", new_uuid, datacenter_id, server_gen)
                self.tmp_actions_were_taken.add(new_uuid)
            self._update_valid_moves_cache(new_uuid)

            self.datacenters_df.loc[
                self.datacenters_df['datacenter_id'] == datacenter_id, 'slots_capacity'] -= slots_required
            self.datacenter_capacities[datacenter_id] -= slots_required

        self.demand_df.loc[self.current_time_step - 1, server_gen] -= num_to_buy
        logger.debug(f"Bought {num_to_buy} servers of type {server_gen} for datacenter {datacenter_id}")

    def _move_server(self, server_uuid, target_dc):
        if server_uuid in self.servers and server_uuid not in self.tmp_actions_were_taken:
            current_dc = self.servers[server_uuid]['datacenter_id']
            slots = self.servers[server_uuid]['slots']
            self.servers[server_uuid]['datacenter_id'] = target_dc

            server_generation = self.servers[server_uuid]['server_type']
            self._log_action("move", server_uuid, target_dc, server_generation)
            self.tmp_actions_were_taken.add(server_uuid)

            self.datacenters_df.loc[self.datacenters_df['datacenter_id'] == current_dc, 'slots_capacity'] += slots
            self.datacenters_df.loc[self.datacenters_df['datacenter_id'] == target_dc, 'slots_capacity'] -= slots
            self.datacenter_capacities[current_dc] += slots
            self.datacenter_capacities[target_dc] -= slots

            self._update_valid_moves_cache(server_uuid)

            logger.debug(f"Moved server {server_uuid} from datacenter {current_dc} to {target_dc}")

    def get_num_servers(self):
        return len(self.servers)

    def _auto_hold_and_dismiss(self, moved_servers):
        to_dismiss = []

        for uuid, server_data in self.servers.items():
            if uuid not in moved_servers and uuid not in self.tmp_actions_were_taken:
                if server_data['remaining_lifespan'] <= 0:
                    to_dismiss.append(uuid)
                else:
                    self._log_action("hold", uuid, server_data['datacenter_id'], server_data['server_type'])
                    self.tmp_actions_were_taken.add(uuid)

        for uuid in to_dismiss:
            if uuid not in self.tmp_actions_were_taken:
                self._dismiss_server(uuid)
                self.tmp_actions_were_taken.add(uuid)

        logger.debug(f"Auto-hold and dismiss processed. Servers to dismiss: {to_dismiss}")

    def _dismiss_server(self, uuid):
        if uuid in self.servers and uuid not in self.tmp_actions_were_taken:
            server_data = self.servers[uuid]
            datacenter_id = server_data['datacenter_id']
            slots_freed = server_data['slots']
            server_generation = server_data['server_type']

            self._log_action("dismiss", uuid, datacenter_id, server_generation)
            self.tmp_actions_were_taken.add(uuid)

            self.datacenters_df.loc[
                self.datacenters_df['datacenter_id'] == datacenter_id, 'slots_capacity'] += slots_freed
            self.datacenter_capacities[datacenter_id] += slots_freed

            del self.servers[uuid]
            if uuid in self.valid_moves_cache:
                del self.valid_moves_cache[uuid]
            logger.debug(f"Dismissed server {uuid} from datacenter {datacenter_id}. Freed {slots_freed} slots.")

    def _log_action(self, action, server_id, datacenter_id, server_generation):
        action_entry = {
            "time_step": self.current_time_step,
            "datacenter_id": datacenter_id,
            "server_generation": server_generation,
            "server_id": server_id,
            "action": action
        }
        self.action_log.append(action_entry)
        logger.debug(f"Action logged: {action_entry}")

    def _check_done(self):
        return self.current_time_step >= 168

    def _can_purchase(self, server_type, datacenter_id, num_to_buy):
        server_gen = self.server_type_map[server_type]
        server_info = self.servers_df[self.servers_df['server_generation'] == server_gen]

        if server_info.empty:
            logger.error(f"No matching server generation found for server_type: {server_gen}")
            return False

        server_info = server_info.iloc[0]
        slots_required = server_info['slots_size'] * num_to_buy

        dc_capacity = self.datacenter_capacities[datacenter_id]
        demand_available = self.demand_df.loc[self.current_time_step - 1, server_gen]
        can_purchase = dc_capacity >= slots_required and num_to_buy <= demand_available
        logger.debug(
            f"Can purchase check for server_type={server_type} (server_generation={server_gen}), datacenter_id={datacenter_id}, num_to_buy={num_to_buy}: {can_purchase}")
        return can_purchase

    def _can_move(self, server_uuid, target_dc):
        if server_uuid not in self.servers:
            logger.error(f"Server UUID {server_uuid} not found for moving.")
            return False
        current_dc = self.servers[server_uuid]['datacenter_id']
        slots_required = self.servers[server_uuid]['slots']
        target_dc_capacity = self.datacenter_capacities[target_dc]
        can_move = target_dc_capacity >= slots_required and current_dc != target_dc
        logger.debug(f"Can move check for server_uuid={server_uuid}, target_dc={target_dc}: {can_move}")
        return can_move

    def _get_uuid_by_index(self, server_index):
        server_uuids = list(self.servers.keys())
        if 0 <= server_index < len(server_uuids):
            uuid = server_uuids[server_index]
            logger.debug(f"UUID found for index {server_index}: {uuid}")
            return uuid
        logger.warning(f"Server index {server_index} is out of range. Total servers: {len(server_uuids)}")
        return None
    def _update_valid_moves_cache(self, server_uuid=None):
        if server_uuid is None:
            self.valid_moves_cache = {}
            for uuid, server_data in self.servers.items():
                self._update_server_valid_moves(uuid, server_data)
        else:
            server_data = self.servers[server_uuid]
            self._update_server_valid_moves(server_uuid, server_data)

    def _update_server_valid_moves(self, uuid, server_data):
        current_dc = server_data['datacenter_id']
        slots = server_data['slots']
        self.valid_moves_cache[uuid] = [
            dc for dc, capacity in self.datacenter_capacities.items()
            if dc != current_dc and capacity >= slots
        ]

    def _get_action_mask(self):
        buy_mask = self._get_buy_mask()
        move_mask = self._get_move_mask()
        return {'buy': buy_mask, 'move': move_mask}

    def _get_buy_mask(self):
        buy_mask = np.zeros((self.n_server_types, self.n_datacenters, 101), dtype=np.int8)

        demand = self.demand_df.iloc[self.current_time_step - 1, 2:].values
        dc_capacities = np.array([self.datacenter_capacities[dc] for dc in self.datacenters_df['datacenter_id']])

        for server_type in range(self.n_server_types):
            server_gen = self.server_type_map[server_type]
            if pd.isna(server_gen):
                continue
            server_info = self.servers_df[self.servers_df['server_generation'] == server_gen].iloc[0]
            slots_per_server = server_info['slots_size']

            max_per_dc = np.minimum(demand[server_type], dc_capacities // slots_per_server)
            max_per_dc = np.minimum(max_per_dc, 100).astype(int)

            for dc_id in range(self.n_datacenters):
                buy_mask[server_type, dc_id, :max_per_dc[dc_id] + 1] = 1

        return buy_mask

    def _get_move_mask(self):
        move_mask = lil_matrix((len(self.servers), self.n_datacenters), dtype=np.int8)
        for i, (uuid, valid_dcs) in enumerate(self.valid_moves_cache.items()):
            if uuid not in self.tmp_actions_were_taken:
                for dc in valid_dcs:
                    move_mask[i, self.dc_id_to_idx[dc]] = 1
        return move_mask

    def step(self, action):
        buy_server_type, buy_dc_idx, num_to_buy, move_server_idx, move_target_dc = action

        # Process buy action
        if num_to_buy > 0:
            datacenter_id = self.idx_to_dc_id.get(buy_dc_idx)
            server_gen = self.server_type_map.get(buy_server_type)
            if datacenter_id is not None and server_gen is not None:
                if self._can_purchase(buy_server_type, datacenter_id, num_to_buy):
                    self._buy_server(server_gen, datacenter_id, num_to_buy)

        # Process move action
        num_servers = self.get_num_servers()
        if num_servers > 0 and move_server_idx < num_servers:
            server_uuid = self._get_uuid_by_index(move_server_idx)
            target_datacenter_id = self.idx_to_dc_id.get(move_target_dc)
            if server_uuid and target_datacenter_id is not None and server_uuid not in self.tmp_actions_were_taken:
                if self._can_move(server_uuid, target_datacenter_id):
                    self._move_server(server_uuid, target_datacenter_id)
                    self.tmp_actions_were_taken.add(server_uuid)

        self._auto_hold_and_dismiss(set())
        self._decrease_lifespan()  # Call the new method here

        done = self._check_done()

        if done:
            self.write_action_log_to_json(f"calculating_{self.current_time_step}.json")
            self.current_score = self._calculate_score(f"calculating_{self.current_time_step}.json")
            logger.info(f"Current score: {self.current_score}, Previous score: {self.previous_score}")

            if self.current_score is not None and self.previous_score is not None:
                reward = self.current_score - self.previous_score
            else:
                reward = self._calculate_reward(self.current_score)

            self.previous_score = self.current_score
        else:
            reward = 0

        self.current_time_step += 1
        next_state = self._get_next_state() if not done else self.state

        # Update action space for the next step
        self.action_space = spaces.MultiDiscrete([
            self.n_server_types,
            self.n_datacenters,
            101,
            max(1, self.get_num_servers()),
            self.n_datacenters
        ])

        info = {}
        return next_state, reward, done, False, info
    def _calculate_score(self, json_filename):
        try:
            demand, datacenters, servers, selling_prices = load_problem_data()
            logger.info(f"Problem data loaded successfully")

            solution = load_solution(json_filename)
            logger.info(f"Solution loaded from {json_filename}")

            score = evaluation_function(solution, demand, datacenters, servers, selling_prices, seed=123)
            logger.info(f"Calculated score: {score}")
            return score
        except Exception as e:
            logger.error(f"Error in _calculate_score: {str(e)}")
            return None


    def _calculate_reward(self, current_score):
        if current_score is None:
            return -1000  # Penalize invalid actions
        elif current_score < 0:
            return current_score / 1e6  # Normalize large negative scores
        else:
            return current_score


    def get_action_log(self):
        return self.action_log

    def _decrease_lifespan(self):
        servers_to_remove = []
        for uuid, server_data in self.servers.items():
            server_data['remaining_lifespan'] -= 1
            if server_data['remaining_lifespan'] <= 0:
                servers_to_remove.append(uuid)

        for uuid in servers_to_remove:
            self._dismiss_server(uuid)

        logger.debug(f"Lifespan decreased for all servers. Removed {len(servers_to_remove)} servers.")
    def write_action_log_to_json(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.action_log, f, indent=2)
        logger.info(f"Action log with {len(self.action_log)} entries written to {filename}")

    def _encode_latency_sensitivity(self, value):
        if value == 'low':
            return 0.0
        elif value == 'medium':
            return 0.5
        elif value == 'high':
            return 1.0
        else:
            raise ValueError(f"Unknown latency sensitivity value: {value}")

    def validate_action_log(self):
        server_actions = {}
        for action in self.action_log:
            server_id = action['server_id']
            time_step = action['time_step']
            action_type = action['action']

            if server_id not in server_actions:
                server_actions[server_id] = []

            server_actions[server_id].append((time_step, action_type))

        for server_id, actions in server_actions.items():
            actions.sort(key=lambda x: x[0])  # Sort by time_step

            if actions[0][1] != 'buy':
                logger.error(f"Server {server_id} first action is not 'buy'")

            for i in range(1, len(actions)):
                if actions[i][0] == actions[i - 1][0]:
                    logger.error(f"Server {server_id} has multiple actions in time step {actions[i][0]}")

                if actions[i][1] == 'buy':
                    logger.error(f"Server {server_id} has 'buy' action after initial purchase")

                if actions[i - 1][1] == 'dismiss' and actions[i][1] != 'buy':
                    logger.error(f"Server {server_id} has action after being dismissed")

        logger.info("Action log validation complete")


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=128)

        n_demand = observation_space['demand'].shape[0]
        n_datacenters = observation_space['datacenters'].shape[0]
        n_servers = observation_space['servers'].shape[0]

        self.demand_net = nn.Sequential(nn.Linear(n_demand, 64), nn.ReLU())
        self.datacenters_net = nn.Sequential(nn.Linear(n_datacenters, 64), nn.ReLU())
        self.servers_net = nn.Sequential(nn.Linear(n_servers, 64), nn.ReLU())

        self.combine_net = nn.Sequential(nn.Linear(64 * 3, 128), nn.ReLU())

    def forward(self, observations):
        demand_features = self.demand_net(observations['demand'])
        datacenters_features = self.datacenters_net(observations['datacenters'])
        servers_features = self.servers_net(observations['servers'])

        combined = torch.cat([demand_features, datacenters_features, servers_features], dim=1)
        return self.combine_net(combined)


def train_and_evaluate():
    # Load data
    demand_df, datacenters_df, servers_df, selling_prices_df = load_problem_data()

    # Create and wrap the environment
    env = make_vec_env(lambda: DataCenterEnv(demand_df, datacenters_df, servers_df, selling_prices_df), n_envs=1)

    # Initialize the model with the custom policy
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    # Training loop
    total_timesteps = 1000000
    for i in range(total_timesteps):
        model.learn(total_timesteps=1, reset_num_timesteps=False)

        if (i + 1) % 1000 == 0:  # Log every 1000 steps
            logger.info(f"Step {i + 1}/{total_timesteps}")

    # Save the trained model
    model.save("datacenter_model")

    # Evaluation loop
    num_episodes = 10
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            if truncated:
                break
        logger.info(f"Episode {episode + 1}, Reward: {episode_reward}")


if __name__ == "__main__":
    train_and_evaluate()
