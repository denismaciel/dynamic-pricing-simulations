# %%
from math import exp, e, factorial, log
import numpy as np


def mean_demand(p):
    a = 20
    alpha = 1
    return a * exp(-alpha * p)


# %%
def optimal_expected_revenue(stock, t):
    a = 20
    lambda_star = a/e
    lambda_star_t = a * (e ** -1)

    def expr(i):
        # return (lambda_star * t) ** i * 1 / factorial(i)
        return (lambda_star_t) ** i * 1 / factorial(i)

    s = (expr(i) for i in range(stock + 1))
    return log(sum(s))


# %%
def optimal_price(stock, t):
    return (
        optimal_expected_revenue(stock, t) - optimal_expected_revenue(stock - 1, t) + 1
    )

# %%
# 1. Set p(stock, remaining_t)
# 2. Set lambda(p)
# 3. Sample duration until next sale
# 4. Update stock and remaining_t, go to 1.

# %%

from collections import namedtuple

sale_record = namedtuple("time_period", "simulation_id t price stock")

stock = 25
total_time = time_left = 200
sale_records = []
while True:
    s = 1
    p = optimal_price(stock, time_left)
    lambda_ = mean_demand(p)
    # Time elapsed until next sale
    time_elapsed = np.random.exponential(lambda_)

    # Season ends before stock is sold out
    if time_left < time_elapsed:
        sale_records.append(sale_record(s, total_time, p, stock))
        break

    stock -= 1
    time_left -= time_elapsed
    sale_records.append(sale_record(s, total_time - time_left, p, stock))

    # Stock sells out before end of season
    if stock == 0:
        break

print(sale_records)
