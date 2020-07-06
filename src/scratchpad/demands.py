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
import numpy as np

def price_strategy(day: int) -> float:
    return 30 if day <= 50 else 60

def demand(p):
    
    def _lamda(p):
        q = 61 - p
        
        if q < 0:
            return 0
        return q
    
    return np.random.poisson(_lamda(p))

prices = [price_strategy(day) for day in range(100)]
demands = [demand(p) for p in prices]

# %%
import matplotlib.pyplot as plt

plt.plot(range(100), demands)

# %% [markdown]
# ## Constant Elasticity
#
# \begin{equation}
# Q = \alpha P^\sigma
# \end{equation}

# %%
P = np.linspace(0, 100, 100)
sigma = -0.8
alpha = 9
Q = alpha * P**sigma
 
plt.plot(Q, P)
plt.xlabel("Quantity")
plt.ylabel("Price")

# %%
