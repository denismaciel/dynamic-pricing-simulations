from __future__ import annotations

from multiprocessing import cpu_count
from multiprocessing import Pool
from typing import Any
from typing import Callable
from typing import NamedTuple

import numpy as np
from dynpric import simulate_market
from dynpric.demand.ferreira2018 import BernoulliDemand
from dynpric.demand.ferreira2018 import PoissonDemand
from dynpric.demand.informs2017 import InformsDemand
from dynpric.firms.ferreira2018 import SamplingStrategies
from dynpric.firms.ferreira2018 import TSFixedFirm
from dynpric.firms.ferreira2018 import TSIngoreInventoryFirm
from dynpric.firms.ferreira2018 import TSUpdateFirm
from dynpric.firms.informs2017 import GreedyFirm
from dynpric.firms.random import RandomFirm
from dynpric.priors import BetaPrior
from dynpric.priors import GammaPrior
from dynpric.types import Belief
from dynpric.types import DemandRealized
from dynpric.types import Firm
from dynpric.types import PriceLevel
from dynpric.types import PricesSet
from dynpric.types import TrialResults


N_TRIALS = 500


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


def clairvoyant_seller_with_bernoulli_demand(trial_id):
    α = 0.25
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
    α = 0.25
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


if __name__ == '__main__':
    import pickle
    import itertools

    class Config(NamedTuple):
        trial_factory: Callable[[int], TrialResults]
        n_periods: int

    def ts_fixed_with_bernoulli(trial_id, n_periods):
        return tsfirm_with_bernoulli_demand(TSFixedFirm, trial_id, n_periods)

    def ts_update_with_bernoulli(trial_id, n_periods):
        return tsfirm_with_bernoulli_demand(TSUpdateFirm, trial_id, n_periods)

    def ts_ignore_inventory_with_bernoulli(trial_id, n_periods):
        return tsfirm_with_bernoulli_demand(
            TSIngoreInventoryFirm, trial_id, n_periods
        )

    factories = [
        ts_fixed_with_bernoulli,
        ts_update_with_bernoulli,
        ts_ignore_inventory_with_bernoulli,
    ]
    n_periods = [10_000]

    configs = sorted(
        [
            Config(trial_factory, n_periods)
            for trial_factory, n_periods in itertools.product(
                factories, n_periods
            )
        ],
        key=lambda config: config.n_periods,
    )
    print(configs)

    def run_simulation(config: Config) -> None:
        results = run_parallel(config.trial_factory, config.n_periods)
        file_name = (
            f'data/{config.trial_factory.__name__}_{config.n_periods}.pickle'
        )
        with open(file_name, 'wb') as f:
            pickle.dump(results, f)

    for config in configs:
        print('Running', config)
        run_simulation(config)
