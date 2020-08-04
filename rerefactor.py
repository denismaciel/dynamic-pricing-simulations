import random
from typing import NamedTuple, List, Callable, Dict, Any, Tuple

from dynpric.seller import Seller
from dynpric.market import Market, Price, Quantity
from dynpric.priors import BetaPrior


class PriceLevel(NamedTuple):
    """
    Container of the information that characterizes the state of a price level
    """

    price: Price
    true_prob: float


PriceLevels = List[PriceLevel]


class Belief(NamedTuple):
    price: Price
    prior: BetaPrior


Beliefs = List[Belief]

price_levels = [
    PriceLevel(29.9, 0.6),
    PriceLevel(34.9, 0.4),
    PriceLevel(39.9, 0.25),
]


def beliefs() -> Beliefs:
    """
    Reset the state of beliefs when initializing a new simulation trial
    """
    return [
        Belief(29.9, BetaPrior(1, 1)),
        Belief(34.9, BetaPrior(1, 1)),
        Belief(39.9, BetaPrior(1, 1)),
    ]


def greedy(b: Belief) -> float:
    return b.prior.expected_value  # type: ignore


def thompson(b: Belief) -> float:
    return b.prior.sample()  # type: ignore


class ThompsonSeller(Seller):
    def __init__(self, beliefs: Beliefs, strategy: Callable[[Belief], float]) -> None:
        self.beliefs = beliefs
        self.strategy = strategy

    def choose_price(self) -> Price:
        profit = [belief.price * self.strategy(belief) for belief in self.beliefs]
        idx = profit.index(max(profit))
        return self.beliefs[idx].price

    def observe_demand(self, q: Quantity, p: Price) -> None:
        # update beliefs with observerd demand
        belief = next(belief for belief in self.beliefs if belief.price == p)
        belief.prior.update(q)


class BinomialMarket(Market):
    def __init__(self, price_levels: PriceLevels) -> None:
        self.price_levels = price_levels

    def realize_demand(self, p: Price) -> Quantity:
        for pl in self.price_levels:
            if pl.price == p:
                return self._simulate_buying_decision(pl)
        else:
            raise ValueError(f"Price {p} is not an allowed price.")

    def _simulate_buying_decision(self, price_level: PriceLevel) -> Quantity:
        if random.random() > price_level.true_prob:
            return 0
        return 1


def simulate(
    S: int,
    T: int,
    initialize_trial: Callable[[], Tuple[Market, Seller]],
    record_state: Callable[[int, Market, Seller, Price, Quantity], Dict],
):
    """
    S: number of trials for the simulation
    T: number of time periods each trial has
    """

    def run_period(t, market, seller, record_state):
        p = seller.choose_price()
        q = market.realize_demand(p)
        seller.observe_demand(q, p)
        return record_state(t, market, seller, p, q)

    trials = []
    for s in range(S):
        if s % 100 == 0:
            print(f"Starting trial number {s}")
        market, seller = initialize_trial()
        periods = [run_period(t, market, seller, record_state) for t in range(T)]
        trials.append({"s": s, "periods": periods})

    return trials


if __name__ == "__main__":

    def thompson_initialize_trial() -> Tuple[Market, Seller]:
        return (
            BinomialMarket(price_levels),
            ThompsonSeller(beliefs(), strategy=thompson),
        )

    def thompson_record_state(
        t: int, market: Market, seller: Seller, p: Price, q: Quantity
    ) -> Dict[str, Any]:
        return {"t": t, "price": p}


    simulation = simulate(100, 100, thompson_initialize_trial, thompson_record_state)

    # Structure of simulation object
    # [
    #   {
    #      "s": 0,
    #      "periods": [
    #          {'t': 0, 'price': 29.99},
    #          {'t': 1, 'price': 34.99},
    #          ...
    #      ]
    #   },
    #   ...
    # ]

    import pandas as pd
    flat = []
    for trial in simulation:
        for period in trial["periods"]:
            flat.append({"s": trial["s"], **period})

    print(pd.DataFrame(flat))
