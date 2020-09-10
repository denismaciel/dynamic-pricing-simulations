# %%
import random
from math import exp, e, factorial, log
from collections import namedtuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def mean_demand(p):
    a = 20
    alpha = 1
    return a * exp(-alpha * p)


# %%
def optimal_expected_revenue(stock, t):
    a = 20
    lambda_star = a / e
    lambda_star_t = a * (e ** -1)

    def expr(i, t):
        return (lambda_star * t) ** i * 1 / factorial(i)
        # return (lambda_star_t) ** i * 1 / factorial(i)

    s = (expr(i, t) for i in range(stock + 1))
    return log(sum(s))


# %%
def optimal_price(stock, t):
    return (
        optimal_expected_revenue(stock, t)
        - optimal_expected_revenue(stock - 1, t)
        + 1
    )


# %%
# ## Approximate Poisson process with a Bernoulli
# 1. Set price p(stock, remaining_t)
# 2. Set lambda(p)
# 3. Sample from a Bernoulli distribution with p = lambda(p) / 1000
# 4. Update remaining_t and stock, which will change only in case of success
# 5. Save record in case of success

# %%

sale_record = namedtuple("time_period", "simulation_id t price stock")
sale_records = []
stock = 25
total_time = 1
PARTITIONS = 50


def bernoulli(p):
    return int(random.random() < p)


for i in range(1, PARTITIONS + 1):
    s = 1
    time_passed = i * total_time / PARTITIONS
    time_left = total_time - time_passed
    print(time_left)

    p = optimal_price(stock, time_left)
    lambda_ = mean_demand(p)

    is_success = bernoulli(lambda_ / PARTITIONS)

    if is_success:
        stock -= 1

        # Stock sells out before end of season
        if stock == 0:
            break
    sale_records.append(sale_record(s, time_passed, p, stock))

sale_records = pd.DataFrame(sale_records)
plt.plot(sale_records["t"], sale_records["price"])
plt.savefig("price_development_bernoulli_approx.png")
plt.close()
# %%
# 1. Set p(stock, remaining_t)
# 2. Set lambda(p)
# 3. Sample duration until next sale
# 4. Update stock and remaining_t, go to 1.

# %%
# sale_record = namedtuple("time_period", "simulation_id t price stock")

# stock = 25
# total_time = time_left = 1
# sale_records = []
# while True:
#     s = 1
#     p = optimal_price(stock, time_left)
#     lambda_ = mean_demand(p)
#     # Time elapsed until next sale
#     time_elapsed = np.random.exponential(lambda_)

#     # Season ends before stock is sold out
#     if time_left < time_elapsed:
#         sale_records.append(sale_record(s, total_time, p, stock))
#         break

#     stock -= 1
#     time_left -= time_elapsed
#     sale_records.append(sale_record(s, total_time - time_left, p, stock))

#     # Stock sells out before end of season
#     if stock == 0:
#         break
