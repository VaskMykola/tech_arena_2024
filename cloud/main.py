from data_preparation import prepare_environment_state

data_path = "../given_info/data"
time_step = 1  # or whatever the current time step is
max_time_steps = 168  # or the total number of time steps in your simulation

env_state = prepare_environment_state(time_step, max_time_steps, data_path)
state_array = env_state.to_numpy()