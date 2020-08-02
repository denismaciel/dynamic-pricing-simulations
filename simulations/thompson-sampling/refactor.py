import random
from typing import List, NamedTuple, Callable

from dynpric.priors import BetaPrior


class PriceLevel(NamedTuple):
    """
    Container of the information that characterizes the state of a price level
    """

    price: float
    true_prob: float
    belief: BetaPrior


PriceLevels = List[PriceLevel]


def price_levels() -> List[PriceLevel]:
    return [
        PriceLevel(29.9, 0.6, BetaPrior(1, 1)),
        PriceLevel(34.9, 0.4, BetaPrior(1, 1)),
        PriceLevel(39.9, 0.25, BetaPrior(1, 1)),
    ]


def true_best(pl: PriceLevel) -> float:
    return pl.true_prob


def greedy(pl: PriceLevel) -> float:
    return pl.belief.expected_value  # type: ignore


def thompson(pl: PriceLevel) -> float:
    return pl.belief.sample()  # type: ignore


def choose_price(
    p_levels: PriceLevels, prob_estimate: Callable[[PriceLevel], float]
) -> PriceLevel:
    profit = [pl.price * prob_estimate(pl) for pl in p_levels]
    idx = profit.index(max(profit))
    return p_levels[idx]


def simulate_buying_decision(price_level: PriceLevel) -> int:
    if random.random() > price_level.true_prob:
        return 0
    return 1


choose_price(price_levels(), thompson)
choose_price(price_levels(), greedy)
choose_price(price_levels(), true_best)


def trial(
    beliefs: PriceLevels, prob_estimate: Callable[[PriceLevel], float]
) -> float:
    pl = choose_price(beliefs, prob_estimate)
    result = simulate_buying_decision(pl)
    pl.belief.update(result)
    return pl.price


def simulate(
    T: int,
    S: int,
    price_levels_constructor: Callable[[], PriceLevels],
    prob_estimate: Callable[[PriceLevel], float],
):
    simulations = []
    for s in range(S):
        if s % 100 == 0:
            print(f"Starting simulation number: {s}")
        beliefs = price_levels()
        timeline = [(t, trial(beliefs, prob_estimate)) for t in range(T)]
        simulations.append(timeline)
    return simulations


output = simulate(10000, 1000, price_levels, thompson)
