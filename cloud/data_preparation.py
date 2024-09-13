# data_preparation.py

import numpy as np
from typing import NamedTuple, Dict
from given_info.utils import load_problem_data
from given_info.evaluation import get_actual_demand


class ServerState(NamedTuple):
    id: str
    generation: str
    capacity: int
    energy_consumption: float
    purchase_price: float
    slots_size: int
    life_expectancy: int
    current_count: int
    can_buy: bool
    demand_high: int
    demand_medium: int
    demand_low: int
    price_high: float
    price_medium: float
    price_low: float


class DatacenterInfo(NamedTuple):
    id: str
    energy_cost: float
    latency_sensitivity: str
    slots_capacity: int
    # current_utilization: float


