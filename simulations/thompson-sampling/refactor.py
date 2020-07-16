import random
from typing import List, NamedTuple, Callable

from dynpric.priors import BetaPrior


class PriceLevel(NamedTuple):
    """
    Container of the information that charcterizes the state of a price level
    """

    price: float
    true_prob: float  # probability a customer makes a purchase at the price level
    belief: BetaPrior


def price_levels() -> List[PriceLevel]:
    return [
        PriceLevel(29.9, 0.3, BetaPrior(1, 1)),
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
    p_levels: List[PriceLevel], prob_estimate: Callable[[PriceLevel], float]
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


beliefs = price_levels()


def simulate(
    T: int,
    S: int,
    price_levels_constructor: Callable[[], List[PriceLevel]],
    prob_estimate: Callable[[PriceLevel], float],
):
    simulations = []
    for s in range(S):
        timeline = []
        for t in range(T):
            pl = choose_price(beliefs, prob_estimate)
            result = simulate_buying_decision(pl)
            pl.belief.update(result)
            timeline.append((t, pl.price))
        simulations.append(timeline)
    return simulations


output = simulate(100, 100, price_levels, thompson)
