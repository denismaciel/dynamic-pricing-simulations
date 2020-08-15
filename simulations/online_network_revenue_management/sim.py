# %%
import random
from typing import List, NamedTuple, Tuple, Iterable, Dict, Any

import numpy as np
from scipy.optimize import linprog
from scipy.optimize.optimize import OptimizeResult

from dynpric.market import Market, Price, Quantity
from dynpric.seller import Seller
from dynpric.priors import BetaPrior
from dynpric.simulation_engine import simulate, trial_factory


# %% [markdown]
# First, we define the containers of information.
#
# - `PriceLevel` is responsible for storing the true probability (`PriceLevel.true_prob`) of one unit being sold under a specific price (`PriceLevel.price`).
# - `Belief` is the analogous of `PriceLevel` but under the view of the seller. Instead of the true probability for a given price, it stores the prior the distribution the seller associates with that price.
#
#
# Notice that all elements of an instance of `PriceLevel` are immutable, since we don't allow for the probability of one unit being sold to change over time. However, `Belief.prior` is mutable, since we're updating it at the end of every period as the seller acquires new information about the demand function. Due to its mutability, we wrap the `Belief`s in the constructor function `beliefs`, so that every trial can start from a clean slate: specifically, the seller starts with a uniform prior for the probability of a unit being sold for every price level. 

# %%
class PriceLevel(NamedTuple):
    """
    Container of the information that characterizes the state of a price level
    """

    price: Price
    true_prob: float


PriceLevels = Iterable[PriceLevel]

price_levels: PriceLevels = (
    PriceLevel(29.9, 0.8),
    PriceLevel(34.9, 0.6),
    PriceLevel(39.9, 0.3),
    PriceLevel(44.9, 0.1),
)


class Belief(NamedTuple):
    price: Price
    prior: BetaPrior


Beliefs = List[Belief]


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


# %%
# def compute_expected_value(
#     price_strategy: Iterable[float],
#     prices: Iterable[float],
#     true_prob: Iterable[float],
# ) -> float:
#     """
#     price_strategy: 
#         probability distribution over allowed set of prices
#     price:
#         vector of allowed prices
#     true_prob:
#         true probabilty of customer buying given a price 
#     """
#     return np.dot(price_strategy, np.array(true_prob) * np.array(prices))

# %% [markdown]
# ## How does the seller behave?
#
# In order to _choose the prices_, the seller:
#
# - Esimates demand
#
# - Chooses optimal price mixture
#
# - Samples on price from the mixture
#
# After the demand is realized, the seller:
#
# - Updates her belief about the demand
# - Updates her inventory

# %% [markdown]
# The first action of the seller is to estimate the demand.

# %%
def find_optimal_price(self, prices, demand) -> OptimizeResult:
    assert len(prices) == len(demand)
    # The reason for the minus sign is that scipy only does minimizations
    objective = [-(p * d) for p, d in zip(prices, demand)]

    # --- Constraints ---

    # 1. Demand is smaller equal than available inventory
    c1 = [demand, self.c]

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


# %%
class _:
    c =  0.25 

demand = [pl.true_prob for pl in price_levels]
price = [pl.price for pl in price_levels]

find_optimal_price(_, price, demand)


# %%
class ConstrainedSeller(Seller):
    
    _find_optimal_price = find_optimal_price
    
    def __init__(self, beliefs, strategy, inventory, n_periods):
        self.beliefs = beliefs
        self.strategy = strategy
        self.inventory = inventory
        self.c = inventory / n_periods

    def choose_price(self):
        demand = self._estimate_demand()
        prices = [belief.price for belief in self.beliefs]

        opt_result = self._find_optimal_price(prices, demand)

        def sample_price(probs, prices) -> float:
            assert len(probs) == len(prices)
            
                
            rounded_probs = np.round(probs, decimals=3)
            if any(p < 0 for p in rounded_probs):
                raise ValueError(rounded_probs)

            normalized_probs = [p / sum(rounded_probs) for p in rounded_probs]
            sampled_price = np.random.choice(prices, size=1, p=normalized_probs)
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


# %%
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


# %%
def greedy(b: Belief) -> float:
    return b.prior.expected_value  # type: ignore

def thompson(b: Belief) -> float:
    return b.prior.sample()  # type: ignore


# %% [markdown]
# ### Simulation Parameters

# %%
N_TRIALS = 500
N_PERIODS = 250
alpha = 0.25
INVENTORY = alpha*N_PERIODS


# %% [markdown]
# ### TS-fixed

# %%
def initialize_trial() -> Tuple[Market, Seller]:
    return (
        BinomialMarket(price_levels),
        ConstrainedSeller(beliefs(), thompson, INVENTORY, N_PERIODS,),
    )

def record_state(
    t: int, market: Market, seller: Seller, p: Price, q: Quantity
) -> Dict[str, Any]:
    
    beliefs = {f"price_{b.price}": b.prior.expected_value for b in seller.beliefs}
    return {"t": t, "price": p, "period_revenue": p * q, **beliefs}


# %%
simulation = simulate(
    S=N_TRIALS,
    T=N_PERIODS,
    trial_runner=trial_factory(initialize_trial, record_state),
)


# %%
def sample_optimal_price(_):
    probs = [0, 0, 0.75, 0.25]
    prices = [29.9, 34.9, 39.9, 44.9]
    return float(np.random.choice(prices, size=1, p=probs))

ConstrainedSeller.choose_price = sample_optimal_price

0.75 * 39.9 * 0.3 + 0.25 * 44.9 * 0.1

# %%
import time

start = time.perf_counter()

simulation = simulate(
    S=N_TRIALS,
    T=N_PERIODS,
    trial_runner=trial_factory(initialize_trial, record_state)
)

end = time.perf_counter()

print("Simulation completed in {}".format(end - start))

# %%
from plotnine import *

from dynpric.simulation_engine import flatten_results
results = flatten_results(simulation)

# %%
from collections import Counter

Counter(results.price)

# %%
revenue = results.groupby('t').period_revenue.mean()
(
    ggplot(aes(revenue.index, revenue))
    + geom_line()
    + lims(y=(5, 20)) 
    + geom_hline(aes(yintercept=10), color='red')
)

# %%
results[['t','price_29.9', 'price_34.9', 'price_39.9', 'price_44.9']].groupby('t').mean().plot()

# %%
counts_per_step = results.groupby(["t", "price"]).size().reset_index(name="n")
counts_per_step["pp"] = counts_per_step["n"] / N_TRIALS

plot = (
    ggplot(counts_per_step, aes("t", "pp", color="factor(price)"))
    + geom_line()
    + labs(title="How often price x was offered in period t averaged across all trials",
           y='%',
           color="Price Levels")
    + lims(y=(0,1))
    + facet_wrap('price') 
)

plot

# %%
boo = counts_per_step["price"] == 44.9

(
    ggplot(counts_per_step[boo], aes("t", "pp"))
    + geom_line()
    + labs(y="Demos", color="A")
)


# %%
counts_per_step.groupby(["price"]).pp.mean()
