# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # Demand Functions
# %% [markdown]
# ## Demand as a Poisson process
#
# For every time $t$, demand is drawn from a Poisson distribution with intensity $\lambda(p)$. As denoted, $\lambda$ depends on price and is assumed to be decreasing, so that the higher the price in period t, the lower the expected demand will be.
#
# To visualize, such a demand function, let's plot 100-day period, where a firm sets $p = 30$ in the first 50 days and $p = 60$ during the final half.
# %%
# %matplotlib inline
# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-deep")


def price_strategy(day: int) -> float:
    return 30 if day <= 50 else 60


def demand(p):
    def _lamda(p):
        q = 100 - p

        if q < 0:
            return 0
        return q

    return np.random.poisson(_lamda(p))


prices = [price_strategy(day) for day in range(100)]
demands = [demand(p) for p in prices]

# %%
plt.plot(demands)
plt.xlabel("Days")
plt.ylabel("Realized Demand")

# %% [markdown]
# Simulation of Gallegos & van Ryzin
#
# \begin{equation}
#     \lambda(p) = ae^{-\alpha p}
# \end{equation}

# %%
from math import exp, e, factorial, log


def mean_demand(p):
    a = 20
    alpha = 1
    return a * exp(-alpha * p)


# %%
def optimal_expected_revenue(stock, t):
    a = 20
    lambda_star = 20 / e
    s = sum(
        (lambda_star * stock) ** i * 1 / factorial(i) for i in range(stock + 1)
    )
    return log(s)


# %%
def optimal_price(stock, t):
    return (
        optimal_expected_revenue(stock, t)
        - optimal_expected_revenue(stock - 1, t)
        + 1
    )


# %%

from collections import namedtuple

time_period = namedtuple("time_period", "simulation_id t demand price stock")

results = []
for s in range(100):
    stock = 25
    timeline = []
    for t in range(1, 101):
        p = optimal_price(stock, 100 - t)
        lambda_ = mean_demand(p)
        d = np.random.poisson(lambda_)

        if stock <= d:  # Ran out of stock
            timeline.append(time_period(s, t, d, p, 0))
            break
        else:
            stock -= d
            timeline.append(time_period(s, t, d, p, stock))

    results.append(pd.DataFrame(timeline))

results = pd.concat(results)

results[["t", "price"]].plot()
plt.savefig("price_development_bernoulli_approx.png")

# %%


fig, (ax1, ax2) = plt.subplots(2, 1)


def add_line(df: pd.DataFrame, ax: mpl.axes.Axes, y: str) -> mpl.axes.Axes:
    return df.plot(
        ax=ax,
        kind="line",
        x="t",
        y=y,
        color="black",
        label=None,
        legend=None,
        alpha=1 / 10,
    )


for key, grp in results.groupby("simulation_id"):
    ax1 = add_line(grp, ax1, "price")
    ax2 = add_line(grp, ax2, "stock")

ax1.set_ylabel("Price")
ax2.set_ylabel("Stock")
ax1.set_xlabel("")
ax1.set_xticklabels([])
fig.tight_layout()
fig.savefig("dflk.png")

# %%
# standanrdize = lambda x: (x - x.mean()) / x.std()

# timeline[['stand_price']] = standanrdize(timeline[['price']])
# timeline[['stand_stock']] = standanrdize(timeline[['stock']]).shift()

# timeline[["price"]].plot()
# plt.savefig("price_stock_dev.png"); plt.close()

# timeline[["stand_price", "stand_stock"]].plot()
# plt.savefig("standardized_price_stock_dev.png"); plt.close()

# %%
optimal_expected_revenue(10, 10)

# %%
mean_demand(1)

# %% [markdown]
# ## Constant Elasticity
#
# \begin{equation}
# Q = \alpha P^\sigma
# \end{equation}

# %%
# P = np.linspace(0, 100, 100)
# sigma = -0.8
# alpha = 9
# Q = alpha * P**sigma

# plt.plot(Q, P)
# plt.xlabel("Quantity")
# plt.ylabel("Price")
