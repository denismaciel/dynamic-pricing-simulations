import random
from typing import NamedTuple, Callable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

beta = 0.99  # discount factor
epsilon = 20  # miminal unitary return
c = 10  # buying cost
n_stores = 2000  # number of stores

from pathlib import Path

figs_folder = Path().absolute() / "figs" / "leloup"


class BetaParams(NamedTuple):
    """
    Parameters of the Beta distribution
    """

    alpha: int
    beta: int


class PriceLevel(NamedTuple):
    """
    Container of the information that charcterizes the state of a price level
    """

    price: int
    true_prob: float  # probability a customer makes a purchase at the price level
    params: BetaParams


def retrieve_price_levels():
    return [
        PriceLevel(30, 1, BetaParams(1, 1)),
        PriceLevel(40, 0.92, BetaParams(1, 1)),
        PriceLevel(50, 0.87, BetaParams(1, 1)),
        PriceLevel(60, 0.65, BetaParams(1, 1)),
        PriceLevel(70, 0.59, BetaParams(1, 1)),
        PriceLevel(80, 0.54, BetaParams(1, 1)),
        PriceLevel(90, 0.51, BetaParams(1, 1)),
        PriceLevel(100, 0.15, BetaParams(1, 1)),
    ]


prices = [pl.price for pl in retrieve_price_levels()]
profits = [pl.price * pl.true_prob - c for pl in retrieve_price_levels()]

plt.plot(prices, profits)
plt.ylabel("Expected Profit")
plt.xlabel("Price")
plt.ylim((0, 100))
plt.xlim((0, 100))
plt.savefig(figs_folder / "expected_profit_vs_price_level.png")
plt.close()

# 1. Find out which price level PL yields the highest expected profit
#     a. If two price levels yield the same profit, pick at random between them.
# 2. Roll the dices  with PL
# 3. Update the beliefs about PL with observed results


def mean_beta(params: BetaParams):
    """
    Given a PriceLevel, returns the mean value of the beta distribution
    """
    return params.alpha / (params.alpha + params.beta)


def calculate_expected_profit(
    pl: PriceLevel,
    expected_probability: Callable[[BetaParams], float],
    cost: float,
):
    return pl.price * expected_probability(pl.params) - cost


def simulate_customer_interaction(pl: PriceLevel):
    """
    For each price level, a customer buys with probability `pl.true_prob`.

    returns 1 if customers buy, returns 0 otherwise.
    """
    if random.random() < pl.true_prob:
        return 1
    return 0


def update_beliefs(pl: PriceLevel, result) -> PriceLevel:
    return PriceLevel(
        pl.price,
        pl.true_prob,
        BetaParams(pl.params.alpha + result, pl.params.beta + (1 - result)),
    )


SIZE = 2000


def run_simulation(
    expected_probability: Callable[[BetaParams], float], size: int
) -> List[List[int]]:
    company_actions = []
    for _ in range(size):
        prices_chosen = []
        price_levels = retrieve_price_levels()
        for _ in range(1000):
            expected_profits = [
                calculate_expected_profit(
                    pl, expected_probability=expected_probability, cost=10
                )
                for pl in price_levels
            ]
            idx = expected_profits.index(max(expected_profits))
            optimal_pl = price_levels[idx]
            result = simulate_customer_interaction(optimal_pl)
            price_levels[idx] = update_beliefs(optimal_pl, result)
            prices_chosen.append(optimal_pl.price)
        company_actions.append(prices_chosen)

    return company_actions


def reshape_simulation_result(
    result: List[List[int]], size: int
) -> pd.DataFrame:

    series = [pd.Series(p_chosen) for p_chosen in result]
    series = pd.concat(series)

    df = pd.DataFrame({"t": series.index, "price_chosen": series})

    pp = df.groupby(["t", "price_chosen"]).size() / size
    pp = pp.reset_index(level="price_chosen", name="pp")
    pp = pp.pivot(columns="price_chosen", values="pp")
    pp[pp.isna()] = 0

    return pp


greedy_result = run_simulation(expected_probability=mean_beta, size=SIZE)

pp = reshape_simulation_result(greedy_result, size=SIZE)
pp = pp[:100]
plt.stackplot(pp.index, pp[50], pp[60], pp[70], pp[80], pp[90], pp[100])
plt.savefig(figs_folder / "greedy_stackplot.png")
plt.close()

pp[[50, 60, 70, 80, 90, 100]].plot()
plt.savefig(figs_folder / "greedy_pp_plot.png")
plt.close()

# Thompson sampling
def sampled_beta(params: BetaParams) -> float:
    return np.random.beta(a=params.alpha, b=params.beta)


thompson_result = run_simulation(expected_probability=sampled_beta, size=SIZE)
pp = reshape_simulation_result(thompson_result, size=SIZE)

plt.stackplot(pp.index, pp[50], pp[60], pp[70], pp[80], pp[90], pp[100])
plt.savefig(figs_folder / "thompson_stackplot.png")
plt.close()

pp[[50, 60, 70, 80, 90, 100]].plot()
plt.savefig(figs_folder / "thompson_pp_plot.png")
plt.close()
