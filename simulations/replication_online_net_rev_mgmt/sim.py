"""
HowTo:
    * Run three simulation with 100, 500 and 1000 periods respectively:
        python3 simulations/replication_online_net_rev_mgmt/sim.py simulate --n-periods 100,500,1000
"""
from __future__ import annotations

import argparse
import itertools
import pathlib
import pickle
import sys
from multiprocessing import cpu_count
from multiprocessing import Pool
from typing import Any
from typing import Callable
from typing import NamedTuple

import click
import numpy as np
from dynpric import simulate_market
from dynpric.demand.ferreira2018 import BernoulliDemand
from dynpric.firms.ferreira2018 import SamplingStrategies
from dynpric.firms.ferreira2018 import TSFixedFirm
from dynpric.firms.ferreira2018 import TSIngoreInventoryFirm
from dynpric.firms.ferreira2018 import TSUpdateFirm
from dynpric.priors import BetaPrior
from dynpric.types import Belief
from dynpric.types import DemandRealized
from dynpric.types import Firm
from dynpric.types import PriceLevel
from dynpric.types import PricesSet
from dynpric.types import TrialResults


ROOT_DIR = pathlib.Path()
N_TRIALS = 500
α = 0.25

# === CLI Interface ===
@click.group()
def cli():
    ...


# === Simulation Code ===
def logger(
    firm: Firm,
    prices_set: PricesSet,
    demand: DemandRealized,
) -> dict[str, Any] | None:

    if (
        type(firm) is TSFixedFirm
        or type(firm) is TSUpdateFirm
        or type(firm) is TSIngoreInventoryFirm
    ):
        return {
            'inventory': firm.inventory,
            'price': prices_set[firm],
            'demand': demand[firm],
            'revenue': prices_set[firm] * demand[firm],
        }
    else:
        return


# Not used
def clairvoyant_seller_with_bernoulli_demand(trial_id):
    N_PERIODS = 1000
    INVENTORY = int(α * N_PERIODS)

    beliefs = [
        Belief(29.9, BetaPrior(1, 1)),
        Belief(34.9, BetaPrior(1, 1)),
        Belief(39.9, BetaPrior(1, 1)),
        Belief(44.9, BetaPrior(1, 1)),
    ]

    TSFixedFirm.price = property(  # type: ignore
        lambda _: np.random.choice(
            [29.9, 34.9, 39.9, 44.9],
            size=1,
            p=[0, 0, 0.75, 0.25],
        )[0]
    )

    ts_update = TSFixedFirm(
        'ts',
        beliefs,
        SamplingStrategies.thompson,
        INVENTORY,
        N_PERIODS,
    )

    demand = BernoulliDemand(
        price_levels=[
            PriceLevel(29.9, 0.8),
            PriceLevel(34.9, 0.6),
            PriceLevel(39.9, 0.3),
            PriceLevel(44.9, 0.1),
        ]
    )

    if trial_id % 2 == 0:
        print(f'Running {trial_id}')

    return simulate_market(
        n_periods=N_PERIODS,
        firms=[ts_update],
        demand=demand,
        logger=logger,
    )


def tsfirm_with_bernoulli_demand(TSFirm, trial_id, n_periods):
    N_PERIODS = n_periods
    INVENTORY = int(α * N_PERIODS)

    beliefs = [
        Belief(29.9, BetaPrior(1, 1)),
        Belief(34.9, BetaPrior(1, 1)),
        Belief(39.9, BetaPrior(1, 1)),
        Belief(44.9, BetaPrior(1, 1)),
    ]

    ts_fixed = TSFirm(
        'ts-firm',
        beliefs,
        SamplingStrategies.thompson,
        INVENTORY,
        N_PERIODS,
    )

    demand = BernoulliDemand(
        price_levels=[
            PriceLevel(29.9, 0.8),
            PriceLevel(34.9, 0.6),
            PriceLevel(39.9, 0.3),
            PriceLevel(44.9, 0.1),
        ]
    )

    if trial_id % 15 == 0:
        print(f'Running {trial_id}')

    return simulate_market(
        n_periods=N_PERIODS,
        firms=[ts_fixed],
        demand=demand,
        logger=logger,
    )


def run_parallel(fn, n_periods):

    with Pool(processes=cpu_count()) as pool:
        args = itertools.product(range(N_TRIALS), [n_periods])
        results = pool.starmap(fn, args)

    return results


def run_sequential(fn, n_periods):
    return [fn(i, n_periods) for i in range(N_TRIALS)]


class Factories:
    def ts_fixed_with_bernoulli(trial_id, n_periods):
        return tsfirm_with_bernoulli_demand(TSFixedFirm, trial_id, n_periods)

    def ts_update_with_bernoulli(trial_id, n_periods):
        return tsfirm_with_bernoulli_demand(TSUpdateFirm, trial_id, n_periods)

    def ts_ignore_inventory_with_bernoulli(trial_id, n_periods):
        return tsfirm_with_bernoulli_demand(
            TSIngoreInventoryFirm, trial_id, n_periods
        )


class Config(NamedTuple):
    trial_factory: Callable[[int], TrialResults]
    n_periods: int

    def __repr__(self):
        return f'{type(self).__name__}({self.trial_factory.__name__}, {self.n_periods})'


def run_simulation(config: Config) -> None:
    def _output_file_name(config):
        return (
            ROOT_DIR
            / 'data'
            / f'{config.trial_factory.__name__}_{config.n_periods}_alpha{α}.pickle'
        )

    results = run_parallel(config.trial_factory, config.n_periods)
    with open(_output_file_name(config), 'wb') as f:
        pickle.dump(results, f)


@cli.command()
@click.option(
    '--n-periods',
    type=str,
    help='Comma-separated number of periods for the trails. For example, 100,500,1000',
)
def simulate(n_periods):
    factories = [
        Factories.ts_fixed_with_bernoulli,
        Factories.ts_update_with_bernoulli,
        Factories.ts_ignore_inventory_with_bernoulli,
    ]

    n_periods = [int(x) for x in n_periods.split(',')]
    configs = sorted(
        [
            Config(trial_factory, n_periods)
            for trial_factory, n_periods in itertools.product(
                factories, n_periods
            )
        ],
        key=lambda config: config.n_periods,
    )

    for config in configs:
        print('Running', config)
        run_simulation(config)


if __name__ == '__main__':
    cli()
