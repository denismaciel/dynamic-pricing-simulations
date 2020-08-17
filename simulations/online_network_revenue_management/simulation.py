# -*- coding: utf-8 -*-
# %% [markdown]
# # Online Network Revenue Management

# %% [markdown]
# Below is the implementation of the TS-Fixed algorithm described in **Ferreira, Kris Johnson, David Simchi-Levi, and He Wang. “Online Network Revenue Management Using Thompson Sampling”**

# %% [markdown]
# # Setup
#
# The setup consists of a seller of one good with limited stock and that can set one of four different prices during a selling season that lasts T periods. The available prices are 29.9, 34.9, 39.9, 44.9. Each price is associated with a demand that is entirely unknown to the seller at the beginning of the season. At each period, the demand can be either 0 or 1 unit. The seller can directly affect the probability that demand turns out to be one by setting the price.
#
# The correspondence between price and demand is the following:
#
# - When the price is 29.9, demand will be 1 with probability 0.8
# - When the price is 34.9, demand will be 1 with probability 0.6
# - When the price is 39.9, demand will be 1 with probability 0.3
# - When the price is 44.9, demand will be 1 with probability 0.1
#
# As one would expect, the higher the price, the lower the expected value of the demand. As already mentioned, this correspondence is unknown to the seller at period t = 0. The way the seller learns about the demand is by setting different prices and observing the resulting demand.
#
# In what follows, we consider the seller to be Bayesian. The seller has the prior belief that the probability for every price is $Beta(1,1)$ distributed.  It is equivalent to a uniform distribution over the interval $[0,1]$. For every available price, the seller assumes that the corresponding probability is equally likely any number between 0 and 1. Thus, the seller beliefs do not incorporate even the commonsense that the higher the price, the lower the demand. According to seller's priors, it is as likely that the price 29.9 corresponds to a probability of 0.01 as the price 44.9 corresponds to a probability of 0.99.
#
# We will see later that, despite this rather unreasonable priors, the seller ends up learning the true demand parameters quite accurately.
#
# Now to the code. We start with the necessary imports.

# %%
import random
import copy
from typing import List, NamedTuple, Tuple, Iterable, Dict, Any

import numpy as np
from scipy.optimize import linprog
from scipy.optimize.optimize import OptimizeResult

from dynpric.market import Market, Price, Quantity
from dynpric.seller import Seller
from dynpric.priors import BetaPrior
from dynpric.simulation_engine import simulate, trial_factory


# %% [markdown]
# Next, we need to decide how to store the necessary information during the simulations.
#
# The prices and the true demand probabilities do not change during the simulation. For that reason, we store them in a named tuple called `PriceLevel`. `PriceLevel.true_prob` gives the probability that demand equals one when price is set to `PriceLevel.price`.
#
# The beliefs of the seller needs to be updated at the end of every period when the seller learns about a new realization of the demand. With that in mind, we store the beliefs in the named tuple `Belief`. It contains two elements: a price, which is a `float`, and an instance of `BetaPrior`. The method `BetaPrior.update` is responsible for the Bayesian updating. It takes the realization of the demand and updates the parameters $\alpha$ and $\beta$ of the Beta distribution.
#
# Also, at the beginning of every trial, the beliefs need to be reset to the priors $Beta(1,1)$. This is accomplished with the function `beliefs`.

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


# %% [markdown]
# ## Seller's optimization problem
#
# The process for the seller to determine which price set comprises three steps:
#
# 1. Esimate the demand
# 2. Given the demand estimates, comupte an optimal price mixture
# 3. Sample one price from the mixture
#
# **Demand estimation**. The first step is to pick a demand probability for each price in order to feed it to the optimization algorithm. This is done via Thompson sampling: the demand probability is sampled from the beta distribution associated with each price. The method `Belief.BetaPrior.sample` implements exactly that. An alternative to come up with the demand probability is to select the expected value of the beta distribution instead of sampling. This is the greedy approach and is exemplified by the function `greedy`.

# %%
def thompson(b: Belief) -> float:
    return b.prior.sample()  # type: ignore

def greedy(b: Belief) -> float:
    return b.prior.expected_value  # type: ignore


# %% [markdown]
# **Optimization**. Given the pairs of prices and demand probabilities, the seller now needs to solve an optimization problem. Ideally, she wants to maximize revenue, but she also must manage a limited inventory. The model specifies that the seller starts with a fixed amount of inventory and is not allowed to replenish it during the sales season. TS-fixed deals with the inventory constraint in a rather static way. During the optimization, we enforce that the resulting expected demand is smaller or equal to a constant $c$, which is the ratio between the inventory and the length of the selling season. $c$ is set at the beginning of the selling season and is **not** updated to reflect the actual development of the inventory. That is the reason why it bears "fixed" in its name.
#
# The optimization result is a vector $x = (x_1, x_2, x_3, x_4)$, the elements of which tell us the probability with which a price level should be chosen.
#
# We formalize the optimization problem below mathematically and in code. For solving it, we use the library scipy. Notice that scipy only does minimizations requires all of the constraints to be specified in the form of "less or equal to." That is the reason for the somewhat unintuitive minus signs used in the objective function and constraints.
#
# \begin{equation}
# LP(d(t)): \max_{x_1, x_2, x_3, x_4} Q = (p_1 q_1) x_1 + (p_2 q_2) x_2 + (p_3 q_3) x_3 + (p_4 q_4) x_4 \\
# \text{subject to } \\
# Q < c \\
# x_1 + x_2 + x_3 + x_4 \leq 1 \\
# x_1, x_2, x_3, x_4 \geq 0
# \end{equation}
#
# <!-- The seller needs to be mindful not to exhaust the full inventory before exploiting the learnings from the exploration.   -->

# %%
def find_optimal_price(self, prices, demand) -> OptimizeResult:
    assert len(prices) == len(demand)
    # The reason for the minus sign is that scipy only does minimizations
    objective = [-(p * d) for p, d in zip(prices, demand)]

    # --- Constraints ---

    # 1. Demand is smaller equal than available inventory
    c1 = [demand, self.c]

    # Sum of probabilities smaller equal one
    c2 = [(1, 1, 1, 1,), 1]

    # 3. Probability of picking a price must be or equal to greater than zero
    c3 = [
        [(-1, 0, 0, 0,), 0],
        [(0, -1, 0, 0,), 0],
        [(0, 0, -1, 0,), 0],
        [(0, 0, 0, -1,), 0],
    ]

    constraints = [c1, c2, *c3]

    lhs_ineq = []
    rhs_ineq = []

    for lhs, rhs in constraints:
        lhs_ineq.append(lhs)
        rhs_ineq.append(rhs)

    opt = linprog(c=objective, A_ub=lhs_ineq, b_ub=rhs_ineq, method="revised simplex")

    return opt


# %% [markdown]
# The solution to the optimization problem of a clairvoyant seller (a seller that knows the actual underlying demand probabilities from the start) is shown below.
#
# We create the `ThrowAwayClass` because `find_optimal_price` is meant to be a method and takes an instance (usually denoted by `self` in Python) as its first argument.  To compute the optimal prices, we need the attribute `c` representing the ratio between the inventory and the periods in the selling season. 
#
# If we set the inventory to 1/4 of the selling season length, the optimal price is to choose \\$ 44.9 with probability 0.25 and \\$ 39.9 with probability 0.75. If we set it to 1/2, the optimal price is to choose \\$39.9 1/3 of the time and \\$34.9 the other 2/3. This illustrates the general tendency that *ceteris paribues*
#
# - if selling season becomes longer, the optimal price increase
# - if the initial stock gets larger, the optimal price decreases

# %%
class ThrowAwayClass0:
    c = 0.25

class ThrowAwayClass1:
    c = 0.5

demand = [pl.true_prob for pl in price_levels]  # true probabilities
price = [pl.price for pl in price_levels]


opt_result = find_optimal_price(ThrowAwayClass0, price, demand)
print("=== Low inventory to duration of selling season ratio ===")
for p, prob in zip(price, opt_result.x):
    print(f"Choose price {p} with probability {round(prob,2)}")


opt_result = find_optimal_price(ThrowAwayClass1, price, demand)
print("\n\n=== High inventory to duration of selling season ratio ===")
for p, prob in zip(price, opt_result.x):
    print(f"Choose price {p} with probability {round(prob,2)}")


# %% [markdown]
# ## Putting everything together
#
# We now have all the elements to construct the `TSFixedSeller` class.
#
# To initialize an instance of the class, we need to specify:
#
# - the beliefs about the demand function, which are beta distributed
# - a strategy to estimate the demand given the current beliefs, which is done via Thompson sampling
# - the initial inventory
# - and the length of the season
#
# With this information, the seller is now ready to interact with the environment. At every period, the interaction takes place via two actions: setting a price (`TSFixedSeller.choose_price`) and observing the realized demand (`TSFixedSeller.observe_demand`). 
#
# The price-setting is the result ofestimating the demand via Thompson sampling (`TSFixedSeller._estimate_demand`), passing the estimated demans through the optimizer (`TSFixedSeller._find_optimal_price`) and finally sampling one price out of the distribution given by the optimizer.
#
# The price is then fed into the `BernoulliMarket`, which simulates the demand by runnning a Bernoulli trial with success probability dependent on the price set.
#
# Finally, the seller observes the realized demand (either 0 or 1) and updates her beliefs about the selected price in a Bayesian fashion.

# %%
class TSFixedSeller(Seller):

    _find_optimal_price = find_optimal_price

    def __init__(self, beliefs, strategy, inventory, n_periods) -> None:
        self.beliefs = beliefs
        self.strategy = strategy
        self.inventory = inventory
        self.c = inventory / n_periods

    def choose_price(self) -> Price:
        demand = self._estimate_demand()
        prices = [belief.price for belief in self.beliefs]

        opt_result = self._find_optimal_price(prices, demand)

        def sample_price(probs, prices) -> float:
            assert len(probs) == len(prices)

            # Ensure probs are always positive
            rounded_probs = np.round(probs, decimals=3)
            if any(p < 0 for p in rounded_probs):
                raise ValueError(rounded_probs)

            # Normalize probs to add up to one
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


class BernoulliMarket(Market):
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


# %% [markdown]
# ## Simulation
#
# ### Parameters
#
# We run 500 trials, each of which consisting of selling season with 500 periods. Inventory is set to 1/4 of the selling season length.
#
# At the end of each period, we record the price, revenue and the belief about the underlying demand of the selller for later analysis.

# %%
N_TRIALS = 500
N_PERIODS = 500
alpha = 0.25
INVENTORY = alpha*N_PERIODS


# %% [markdown]
# ### TS-fixed
#
# Run the simulation for `TSFixerSeller`.
#
# Note that the trials might finish out of order. It happens because they are running in multiple processes in order to take advantage of all the CPU cores of the computer.

# %%
def initialize_trial() -> Tuple[Market, Seller]:
    return (
        BernoulliMarket(price_levels),
        TSFixedSeller(beliefs(), thompson, INVENTORY, N_PERIODS,),
    )

def record_state(
    t: int, market: Market, seller: Seller, p: Price, q: Quantity
) -> Dict[str, Any]:
    
    beliefs = {f"price_{b.price}": b.prior.expected_value for b in seller.beliefs}
    return {"t": t, "price": p, "period_revenue": p * q, **beliefs}

ts_fixed = simulate(
    S=N_TRIALS,
    T=N_PERIODS,
    trial_runner=trial_factory(initialize_trial, record_state),
)

from dynpric.simulation_engine import flatten_results

flatten_results(ts_fixed).to_parquet(f"data/ts_fixed_trials{N_TRIALS}_periods{N_PERIODS}.parquet")


# %% [markdown]
# ### Clairvoyant Seller

# %% [markdown]
# We simulate the same setting for a clairvoyant seller. In fact, we can even reuse all of `TSFixedSeller` functionality only change the `choose_price` method by enforcing it to always return the optimal price mixture.

# %%
def sample_optimal_price(_):
    probs = [0, 0, 0.75, 0.25]
    prices = [29.9, 34.9, 39.9, 44.9]
    return float(np.random.choice(prices, size=1, p=probs))


ClairvoyantSeller = copy.deepcopy(TSFixedSeller)
ClairvoyantSeller.choose_price = sample_optimal_price


def clairvoyant_initialize_trial() -> Tuple[Market, Seller]:
    return (
        BernoulliMarket(price_levels),
        ClairvoyantSeller(beliefs(), thompson, INVENTORY, N_PERIODS,),
    )


clairvoyant = simulate(
    S=N_TRIALS,
    T=N_PERIODS,
    trial_runner=trial_factory(clairvoyant_initialize_trial, record_state),
)

flatten_results(clairvoyant).to_parquet(
    f"data/clairvoyant_trials{N_TRIALS}_periods{N_PERIODS}.parquet"
)
