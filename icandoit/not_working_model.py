import traceback

import gymnasium
from gymnasium import spaces
import numpy as np
import pandas as pd
import uuid
import json
from scipy.sparse import lil_matrix
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

        self.max_servers = 10000  # Maximum number of servers allowed

        # Initialize dynamic purchase limits
        self.dynamic_purchase_limits = self._initialize_purchase_limits()
        max_purchase_limit = max(max(limits.values()) for limits in self.dynamic_purchase_limits.values())

        self.action_space = spaces.MultiDiscrete([
            self.n_server_types,
            self.n_datacenters,
            max_purchase_limit + 1,
            self.max_servers,
            self.n_datacenters
        ])

        self.observation_space = spaces.Dict({
            'demand': spaces.Box(low=0, high=np.inf, shape=(self.n_server_types,), dtype=np.int32),
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

    def _initialize_purchase_limits(self):
        # This method should return a dictionary of purchase limits for each server type and time step
        # The structure could be: {time_step: {server_type: max_purchase}}
        # You'll need to implement this based on your database structure
        # For now, we'll use a dummy implementation
        limits = {}
        for t in range(1, 169):  # Assuming 168 time steps
            limits[t] = {server_gen: 100 for server_gen in self.server_type_map.values()}
        return limits

    @property
    def current_server_count(self):
        return len(self.servers)

    def reset(self):
        self.current_time_step = 1
        self.servers = {}
        self.action_log = []
        self.state = self._get_initial_state()
        self.current_score = None
        self.valid_moves_cache = {}
        self.datacenter_capacities = {dc: capacity for dc, capacity in
                                      zip(self.datacenters_df['datacenter_id'], self.datacenters_df['slots_capacity'])}
        self.tmp_actions_were_taken = set()
        self._update_purchase_limits()

        logger.info(f"Environment reset at time_step {self.current_time_step}")

        state_array = np.concatenate([
            self.state['demand'],
            self.state['datacenters'],
            self.state['servers']
        ])

        return state_array

    def _update_purchase_limits(self):
        self.current_purchase_limits = self.dynamic_purchase_limits.get(self.current_time_step, {})

    def step(self, action):
        logger.debug(f"Step started at time {self.current_time_step}")
        logger.debug(f"Action received: {action}")
        reward = 0

        self._decrease_lifespan()
        self.tmp_actions_were_taken.clear()
        self._update_purchase_limits()

        buy_server_type, buy_dc_idx, num_to_buy, move_server_idx, move_target_dc = action

        datacenter_id = self.idx_to_dc_id.get(buy_dc_idx)
        server_gen = self.server_type_map.get(buy_server_type)
        if datacenter_id is not None and server_gen is not None:
            logger.debug(
                f"Processing buy action: server_type={server_gen}, datacenter_id={datacenter_id}, num_to_buy={num_to_buy}")
            if num_to_buy > 0 and self._can_purchase(buy_server_type, datacenter_id, num_to_buy):
                self._buy_server(server_gen, datacenter_id, num_to_buy)
            else:
                self._log_action("buy_attempt", f"attempt_{self.current_time_step}", datacenter_id, server_gen)

        moved_servers = set()
        if move_server_idx < self.current_server_count:
            server_uuid = self._get_uuid_by_index(move_server_idx)
            target_datacenter_id = self.idx_to_dc_id.get(move_target_dc)
            if server_uuid and target_datacenter_id is not None and server_uuid not in self.tmp_actions_were_taken:
                logger.debug(
                    f"Processing move action: server_uuid={server_uuid}, target_datacenter_id={target_datacenter_id}")
                if self._can_move(server_uuid, target_datacenter_id):
                    self._move_server(server_uuid, target_datacenter_id)
                    moved_servers.add(server_uuid)
                    self.tmp_actions_were_taken.add(server_uuid)

        self._auto_hold_and_dismiss(moved_servers)

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

        self.current_time_step += 1

        next_state = self._get_next_state() if not done else self.state
        next_state_array = np.concatenate([
            next_state['demand'],
            next_state['datacenters'],
            next_state['servers']
        ])

        return next_state_array, reward, done, {}

    def _decrease_lifespan(self):
        for uuid, server_data in self.servers.items():
            server_data['remaining_lifespan'] -= 1
        logger.debug(f"Lifespan decreased for all servers. Current server states: {self.servers}")

    def _get_initial_state(self):
        demand = self.demand_df.iloc[self.current_time_step - 1, 2:].values
        datacenters = self.datacenters_df[['slots_capacity', 'cost_of_energy', 'latency_sensitivity']].values.flatten()
        servers = np.zeros((self.n_datacenters * self.n_server_types,))
        logger.debug(f"Initial state set with demand: {demand}, datacenters: {datacenters}, servers: {servers}")
        return {'demand': demand, 'datacenters': datacenters, 'servers': servers}

    def _get_next_state(self):
        demand = self.demand_df.iloc[self.current_time_step, 2:].values
        datacenters = self.datacenters_df[['slots_capacity', 'cost_of_energy', 'latency_sensitivity']].values.flatten()
        servers = np.zeros((self.n_datacenters * self.n_server_types,))
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
            self._log_action("buy", new_uuid, datacenter_id, server_gen)
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
        max_purchase = self.current_purchase_limits.get(server_gen, 0)

        if num_to_buy > max_purchase:
            logger.debug(f"Cannot purchase {num_to_buy} servers of type {server_gen}. Maximum allowed: {max_purchase}")
            return False

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
        if server_index < len(self.servers):
            uuid = list(self.servers.keys())[server_index]
            logger.debug(f"UUID found for index {server_index}: {uuid}")
            return uuid
        logger.error(f"Server index {server_index} is out of range.")
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
        server_selection_mask = np.zeros(self.max_servers, dtype=np.int8)
        server_selection_mask[:self.current_server_count] = 1
        return {
            'buy': buy_mask,
            'move': move_mask,
            'server_selection': server_selection_mask
        }

    def _get_buy_mask(self):
        buy_mask = np.zeros((self.n_server_types, self.n_datacenters, self.action_space.nvec[2]), dtype=np.int8)

        demand = self.demand_df.iloc[self.current_time_step - 1, 2:].values
        dc_capacities = np.array([self.datacenter_capacities[dc] for dc in self.datacenters_df['datacenter_id']])

        for server_type in range(self.n_server_types):
            server_gen = self.server_type_map[server_type]
            if pd.isna(server_gen):
                continue
            server_info = self.servers_df[self.servers_df['server_generation'] == server_gen].iloc[0]
            slots_per_server = server_info['slots_size']

            max_purchase = self.current_purchase_limits.get(server_gen, 0)
            max_per_dc = np.minimum(demand[server_type], dc_capacities // slots_per_server)
            max_per_dc = np.minimum(max_per_dc, max_purchase).astype(int)

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

    def _calculate_score(self, json_filename):
        try:
            demand, datacenters, servers, selling_prices = load_problem_data()
            logger.info(f"Problem data loaded successfully")

            solution = load_solution(json_filename)
            logger.info(f"Solution loaded from {json_filename}")

            score = evaluation_function(solution, demand, datacenters, servers, selling_prices, seed=123)
            if score is None:
                logger.error(
                    "Evaluation function returned None. This might indicate an issue with the solution format or content.")
                logger.error(f"First few entries of the solution: {solution[:5]}")
                logger.error(f"Number of entries in the solution: {len(solution)}")
            else:
                logger.info(f"Calculated score: {score}")
            return score
        except Exception as e:
            logger.error(f"Error in _calculate_score: {str(e)}")
            logger.error(f"Error traceback: {traceback.format_exc()}")
            return None
    def _calculate_reward(self, current_score):
        if current_score is None:
            return -1000  # Penalize invalid actions
        elif current_score < 0:
            return current_score / 1e6  # Normalize large negative scores
        else:
            return current_score  # Positive scores are good, keep them as is

    def get_action_log(self):
        return self.action_log

    def write_action_log_to_json(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.action_log, f, indent=2)
        logger.info(f"Action log with {len(self.action_log)} entries written to {filename}")

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

            if actions[0][1] not in ['buy', 'buy_attempt']:
                logger.error(f"Server {server_id} first action is not 'buy' or 'buy_attempt'")

            for i in range(1, len(actions)):
                if actions[i][0] == actions[i - 1][0]:
                    logger.error(f"Server {server_id} has multiple actions in time step {actions[i][0]}")

                if actions[i][1] == 'buy' and actions[i - 1][1] != 'buy_attempt':
                    logger.error(f"Server {server_id} has 'buy' action after non-buy_attempt")

                if actions[i - 1][1] == 'dismiss' and actions[i][1] not in ['buy', 'buy_attempt']:
                    logger.error(f"Server {server_id} has action after being dismissed")

        logger.info("Action log validation complete")
if __name__ == "__main__":
    # Load data
    demand_df, datacenters_df, servers_df, selling_prices_df = load_problem_data(r"../given_info/data")

    # Create environment
    env = DataCenterEnv(demand_df, datacenters_df, servers_df, selling_prices_df)

    # Run simulation
    obs = env.reset()
    done = False
    total_reward = 0

    for time_step in range(1, 169):  # Ensure we go through all 168 time steps
        logger.info(f"Starting time step {time_step}")

        # Generate random action with higher probability of buying
        buy_server_type = np.random.randint(env.n_server_types)
        buy_dc_idx = np.random.randint(env.n_datacenters)
        num_to_buy = np.random.randint(11)  # 0 to 10, increased probability of non-zero
        move_server_idx = np.random.randint(env.max_servers)
        move_target_dc = np.random.randint(env.n_datacenters)

        action = [buy_server_type, buy_dc_idx, num_to_buy, move_server_idx, move_target_dc]

        # ... (rest of the loop remains the same)

        # Take step in environment
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        logger.info(f"Time step: {env.current_time_step}, Reward: {reward}, Total Reward: {total_reward}")
        logger.debug(f"Current action log length: {len(env.action_log)}")

        if done:
            break

    logger.info(f"Simulation completed. Total actions logged: {len(env.action_log)}")

    # Validate and write action log to JSON file
    env.validate_action_log()

    # Print sample of action log
    logger.info(f"Sample of action log (first 10 entries):")
    for action in env.action_log[:10]:
        logger.info(str(action))

    # Evaluate the solution
    json_filename = 'solution.json'
    if len(env.action_log) > 0:
        env.write_action_log_to_json(json_filename)
        logger.info(f"Action log written to {json_filename}")

        logger.info("Starting score calculation...")
        score = env._calculate_score(json_filename)
        logger.info(f"Score calculation completed. Result: {score}")

        if score is not None:
            logger.info(f"Score achieved: {score}.")
            if score > 0:
                logger.info("Positive score achieved. Saving solution.")
                env.write_action_log_to_json(f'solution_score_{score:.2f}.json')
            else:
                logger.info("Score is not positive.")
        else:
            logger.error("Failed to calculate score.")
    else:
        logger.error("Action log is empty. No file written.")

    logger.info(f"Final action log length: {len(env.action_log)}")