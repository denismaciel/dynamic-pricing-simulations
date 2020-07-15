import random
from typing import List, NamedTuple, Tuple, Iterable

import numpy as np
from scipy.optimize import linprog
from scipy.optimize.optimize import OptimizeResult


demand = (0.8, 0.6, 0.3, 0.1)


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

    price: float
    true_prob: float  # probability a customer makes a purchase at the price level
    params: BetaParams


def retrieve_price_levels() -> List[PriceLevel]:
    return [
        PriceLevel(29.9, 0.8, BetaParams(1, 1)),
        PriceLevel(34.9, 0.6, BetaParams(1, 1)),
        PriceLevel(39.9, 0.3, BetaParams(1, 1)),
        PriceLevel(44.9, 0.1, BetaParams(1, 1)),
    ]


def find_optimal_price(demand: List[float]) -> OptimizeResult:
    prices = (29.9, 34.9, 39.9, 44.9)
    alpha = 0.6
    T = 10_000
    inventory = alpha * T
    c = inventory / T

    assert len(prices) == len(demand)
    # The reason for the minus sign is that scipy only does minimizations
    objective = [-(p * d) for p, d in zip(prices, demand)]

    # --- Constraints ---:

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

    opt = linprog(c=objective, A_ub=lhs_ineq, b_ub=rhs_ineq, method="revised simplex")

    return opt


def get(attr: str, price_levels: List[PriceLevel]) -> List[float]:
    return [getattr(x, attr) for x in price_levels]


def sample_beta(params: BetaParams) -> float:
    return np.random.beta(a=params.alpha, b=params.beta)


def sample_implemented_price(
    opt_result: OptimizeResult, price_levels: List[PriceLevel]
) -> Tuple[int, float]:
    assert len(opt_result.x) == len(price_levels)

    prices = get("price", price_levels)
    probabilities = [np.round(p, 5) for p in opt_result.x]

    prob_p_inf = 1 - sum(probabilities)

    prices.append(np.inf)
    probabilities.append(prob_p_inf)

    price = np.random.choice(prices, 1, p=probabilities)

    return prices.index(price), price


def sample_demand(price_level: PriceLevel) -> int:
    return 1 if price_level.true_prob > random.random() else 0


def update_beliefs(pl: PriceLevel, result: int) -> PriceLevel:
    return PriceLevel(
        pl.price,
        pl.true_prob,
        BetaParams(pl.params.alpha + result, pl.params.beta + (1 - result)),
    )


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


price_levels = retrieve_price_levels()
compute_expected_value(
    [1, 0, 0, 0], get("price", price_levels), get("true_prob", price_levels)
)

# Expected value of optimal price strategy (i.e. when the seller knows true
# parameters of demand)

opt = find_optimal_price(get('true_prob', price_levels))
optimal_prices = opt.x
PRICES = get("price", price_levels)
TRUE_PROBS = get("true_prob", price_levels)
UPPER_BOUND = compute_expected_value(optimal_prices, PRICES, TRUE_PROBS) 

if __name__ == "__main__":
    for s in range(1):
        price_levels = retrieve_price_levels()
        for _ in range(10000):
            sampled_params = [sample_beta(pl.params) for pl in price_levels]
            opt_result = find_optimal_price(sampled_params)
            # print(list(opt_result.x))
            idx, price = sample_implemented_price(opt_result, price_levels)

            if price == np.inf:
                ...
            else:
                print(compute_expected_value(opt_result.x, PRICES, TRUE_PROBS))
                # print(list(opt_result.x))
                print([pl.params.alpha / (pl.params.alpha + pl.params.beta) for pl in price_levels])
                d = sample_demand(price_levels[idx])
                print(d)
                price_levels[idx] = update_beliefs(price_levels[idx], d)

for pl in price_levels:
    print(pl.price, pl.params.alpha + pl.params.beta)

