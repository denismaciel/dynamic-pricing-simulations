import multiprocessing
import os
from itertools import product
from functools import partial
from typing import Callable, Dict, Tuple

from dynpric.seller import Seller
from dynpric.market import Market, Price, Quantity


def run_period(t, market, seller, record_state):
    p = seller.choose_price()
    q = market.realize_demand(p)
    seller.observe_demand(q, p)
    return record_state(t, market, seller, p, q)


def run_trial(
    s,
    T,
    reporting_frequency,
    initializer: Callable[[], Tuple[Market, Seller]],
    state_recorder: Callable[[int, Market, Seller, Price, Quantity], Dict],
) -> Dict:
    market, seller = initializer()
    periods = [run_period(t, market, seller, state_recorder) for t in range(T)]
    if s % reporting_frequency == 0:
        print(f"Finished trial number {s}")
    return {"s": s, "periods": periods}


def trial_factory(
    initializer: Callable[[], Tuple[Market, Seller]],
    state_recorder: Callable[[int, Market, Seller, Price, Quantity], Dict],
) -> Callable[[int, int], Dict]:
    return partial(run_trial, initializer=initializer, state_recorder=state_recorder)


def simulate(S: int, T: int, trial_runner: Callable, execution_mode='parallel'):
    """
    S: number of trials for the simulation
    T: number of time periods each trial has
    """
    print("Starting simulation...")

    REPORTING_FREQUENCY = S // 10

    if execution_mode == 'parallel':
        pool = multiprocessing.Pool(processes=os.cpu_count() - 1)
        return pool.starmap(trial_runner, product(range(S), [T], [REPORTING_FREQUENCY]))
    elif execution_mode == 'sequential':
        return [trial_runner(s, T, REPORTING_FREQUENCY) for s in range(S)]
