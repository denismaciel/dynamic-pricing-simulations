from __future__ import annotations

import argparse
import functools
import itertools
import pathlib
import pickle
from multiprocessing import cpu_count
from multiprocessing import Pool
from typing import Any

import dynpric.firms.ferreira2018 as firms
import pandas as pd
from dynpric import simulate_market
from dynpric.demand.informs2017 import TrialDemand
from dynpric.priors import GammaPrior
from dynpric.types import Belief
from dynpric.types import DemandRealized
from dynpric.types import Firm
from dynpric.types import PricesSet

from simulations.exploration_exploitation.sim import GreedySeller
from simulations.exploration_exploitation.sim import εGreedySeller

DATA_DIR = pathlib.Path(__file__).parent / 'data'
# e-greedy vs thompson sampling
# thompson sampling vs thompson sampling
# Come up with a demand
# Poisson

# === CLI ===
COMMANDS = {}


def command(fn):
    COMMANDS[fn.__name__] = fn
    return fn


# === Simulation ===
α = 0.25
N_PERIODS = 1000
UNITS_PER_PERIOD = 100
INVENTORY = int(α * N_PERIODS * UNITS_PER_PERIOD)
N_TRIALS = 500


experiments = {
    'ts_fixed_vs_egreedy': {
        'firms': [
            {'name': 'εGreedy', 'type': 'εgreedy'},
            {'name': 'TsFixed', 'type': 'ts_fixed'},
        ]
    },
    'ts_fixed_vs_itself': {
        'firms': [
            {'name': 'TsFixedA', 'type': 'ts_fixed'},
            {'name': 'TsFixedB', 'type': 'ts_fixed'},
        ]
    },
    'ts_fixed_vs_greedy': {
        'firms': [
            {'name': 'TsFixed', 'type': 'ts_fixed'},
            {'name': 'Greedy', 'type': 'greedy'},
        ]
    },
    'egreedy_vs_greedy': {
        'firms': [
            {'name': 'εGreedy', 'type': 'εgreedy'},
            {'name': 'Greedy', 'type': 'greedy'},
        ]
    },
    'all': {
        'firms': [
            {'name': 'εGreedy', 'type': 'εgreedy'},
            {'name': 'Greedy', 'type': 'greedy'},
            {'name': 'TsFixed', 'type': 'ts_fixed'},
            {'name': 'TsUpdate', 'type': 'ts_update'},
        ]
    },
    'ts_update_vs_greedy': {
        'firms': [
            {'name': 'TsUpdate', 'type': 'ts_update'},
            {'name': 'Greedy', 'type': 'greedy'},
        ]
    },
}

EXPERIMENT_NAME = 'ts_update_vs_greedy'
assert EXPERIMENT_NAME in experiments


def logger(
    firm: Firm,
    prices_set: PricesSet,
    demand: DemandRealized,
) -> dict[str, Any] | None:
    return {
        'price': prices_set[firm],
        'demand': demand[firm],
        'revenue': prices_set[firm] * demand[firm],
    }


def beliefs():
    return [
        Belief(29.9, GammaPrior(50, 1)),
        Belief(34.9, GammaPrior(50, 1)),
        Belief(39.9, GammaPrior(50, 1)),
        Belief(44.9, GammaPrior(50, 1)),
    ]


class Firms:
    @staticmethod
    def εgreedy(name):
        return εGreedySeller(name, beliefs(), 0.5)

    @staticmethod
    def greedy(name):
        return GreedySeller(name, beliefs())

    @staticmethod
    def ts_fixed(name):
        return firms.TSFixedFirm(
            name,
            beliefs(),
            firms.SamplingStrategies.thompson,
            INVENTORY,
            N_PERIODS,
        )

    @staticmethod
    def ts_update(name):
        return firms.TSUpdateFirm(
            name,
            beliefs(),
            firms.SamplingStrategies.thompson,
            INVENTORY,
            N_PERIODS,
        )


def run_parallel(fn, n_periods):
    with Pool(processes=cpu_count()) as pool:
        args = itertools.product(range(N_TRIALS), [n_periods])
        results = pool.starmap(fn, args)

    return results


def _log(trial_id: int) -> None:
    if trial_id % 25 == 0:
        print('Running', trial_id)


def instatiate_firms(firms_spec):
    return [getattr(Firms, spec['type'])(spec['name']) for spec in firms_spec]


def trial_factory(trial_id, n_periods, firms_spec: list[dict[str, str]]):

    _log(trial_id)

    demand = TrialDemand(λ=100, θ_sho=0.5, θ_loy=0.5, β_loy=40, β_sho=25)
    return simulate_market(
        n_periods,
        firms=instatiate_firms(firms_spec),
        demand=demand,
        logger=logger,
    )


@command
def simulate():
    experiment = experiments[EXPERIMENT_NAME]
    speced_trial_factory = functools.partial(
        trial_factory, firms_spec=experiment['firms']
    )
    results = run_parallel(speced_trial_factory, N_PERIODS)

    with open(DATA_DIR / f'{EXPERIMENT_NAME}.pickle', 'wb') as f:
        pickle.dump(results, f)


@command
def generate_csv():
    def process_trial(trial, trial_id):
        logs, history = trial

        unpacked_trial = []
        for period_id, period in enumerate(logs):
            unpacked_trial.extend(process_period(period, trial_id, period_id))
        return unpacked_trial

    def process_period(period, trial_id, period_id):
        return (
            {
                'trial_id': trial_id,
                'period_id': period_id,
                'firm': firm.name,
                **info,
            }
            for firm, info in period.items()
        )

    def process_pickle(file):
        print('Processing', file)

        with open(pkl, 'rb') as f:
            results = pickle.load(f)

        df = pd.concat(
            [
                pd.DataFrame(process_trial(trial, trial_id))
                for trial_id, trial in enumerate(results)
            ]
        )
        print(df[-10:])
        print(df.shape)
        df.to_csv(DATA_DIR / f'{file.stem}.csv', index=False)

    for pkl in DATA_DIR.glob('*.pickle'):
        process_pickle(pkl)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command')
    args = parser.parse_args()
    COMMANDS[args.command]()


if __name__ == '__main__':
    main()
