import random
from typing import List, NamedTuple, Tuple, Iterable, Dict, Any

import numpy as np
from scipy.optimize import linprog
from scipy.optimize.optimize import OptimizeResult

from dynpric.market import Market, Price, Quantity
from dynpric.seller import Seller
from dynpric.priors import BetaPrior
from dynpric.simulation_engine import simulate, trial_factory


class PriceLevel(NamedTuple):
    """
    Container of the information that characterizes the state of a price level
    """

    price: Price
    true_prob: float


class Belief(NamedTuple):
    price: Price
    prior: BetaPrior


Beliefs = List[Belief]

price_levels = [
    PriceLevel(29.9, 0.8),
    PriceLevel(34.9, 0.6),
    PriceLevel(39.9, 0.3),
    PriceLevel(44.9, 0.1),
]

PriceLevels = List[PriceLevel]


def beliefs() -> Beliefs:
    """
    Reset the state of beliefs when initializing a new simulation trial
    """
    return [
        Belief(29.9, BetaPrior(1, 1)),
        Belief(34.9, BetaPrior(1, 1)),
        Belief(39.9, BetaPrior(1, 1)),
        Belief(44.9, BetaPrior(1, 1)),
    ]


# demand = (0.8, 0.6, 0.3, 0.1)
# prices = (29.9, 34.9, 39.9, 44.9)


def compute_expected_value(
    price_strategy: Iterable[float],
    prices: Iterable[float],
    true_prob: Iterable[float],
) -> float:
    """
    price_strategy: 
        probability distribution over allowed set of prices
    price:
        vector of allowed prices
    true_prob:
        true probabilty of customer buying given a price 
    """
    return np.dot(price_strategy, np.array(true_prob) * np.array(prices))


class ConstrainedSeller(Seller):
    def __init__(self, beliefs, strategy):
        self.beliefs = beliefs
        self.strategy = strategy

    def choose_price(self):
        demand = self._estimate_demand()
        prices = [belief.price for belief in self.beliefs]

        opt_result = self._find_optimal_price(prices, demand)

        def sample_price(probs, prices) -> float:
            assert len(probs) == len(prices)
            normalized_probs = [p / sum(probs) for p in probs]
            sampled_price = np.random.choice(prices, 1, normalized_probs)
            return float(sampled_price)

        chosen_price = sample_price(opt_result.x, prices)

        return chosen_price

    def observe_demand(self, q: Quantity, p: Price) -> None:
        # update beliefs with observerd demand
        belief = next(belief for belief in self.beliefs if belief.price == p)
        belief.prior.update(q)

    def _estimate_demand(self) -> List[int]:
        demand = [self.strategy(belief) for belief in self.beliefs]
        return demand

    def _find_optimal_price(self, prices, demand) -> OptimizeResult:
        """
        Returns a vector with th optimal probability distribution over
        the prices
        """
        alpha = 0.6
        T = 10_000
        inventory = alpha * T
        c = inventory / T

        assert len(prices) == len(demand)
        # The reason for the minus sign is that scipy only does minimizations
        objective = [-(p * d) for p, d in zip(prices, demand)]

        # --- Constraints ---

        # 1. Demand is smaller equal than available inventory
        c1 = [demand, c]

        # Sum of probabilities smaller equal zero
        c2 = [(1, 1, 1, 1,), 1]

        # 3. Probability of picking a price must be greater than zero
        c3 = [(-1, -1, -1, -1,), 0]

        constraints = [c1, c2, c3]

        lhs_ineq = []
        rhs_ineq = []

        for lhs, rhs in constraints:
            lhs_ineq.append(lhs)
            rhs_ineq.append(rhs)

        opt = linprog(
            c=objective, A_ub=lhs_ineq, b_ub=rhs_ineq, method="revised simplex"
        )

        return opt


class BinomialMarket(Market):
    def __init__(self, price_levels: PriceLevels) -> None:
        self.price_levels = price_levels

    def _simulate_buying_decision(self, price_level: PriceLevel) -> Quantity:
        if random.random() > price_level.true_prob:
            return 0
        return 1

    def realize_demand(self, p: Price) -> Quantity:
        for pl in self.price_levels:
            if pl.price == p:
                return self._simulate_buying_decision(pl)
        else:
            raise ValueError(f"Price {p} is not an allowed price.")


def greedy(b: Belief) -> float:
    return b.prior.expected_value  # type: ignore


def thompson(b: Belief) -> float:
    return b.prior.sample()  # type: ignore


def initialize_trial() -> Tuple[Market, Seller]:
    return (
        BinomialMarket(price_levels),
        ConstrainedSeller(beliefs(), greedy),
    )


def record_state(
    t: int, market: Market, seller: Seller, p: Price, q: Quantity
) -> Dict[str, Any]:
    return {"t": t, "price": p}


simulation = simulate(
    S=100,
    T=100,
    trial_runner=trial_factory(initialize_trial, record_state),
    execution_mode="sequential",
)
