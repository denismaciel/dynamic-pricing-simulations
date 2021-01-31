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
# ## Dynamic pricing as a dynamic programming problem
#
# ## Intro
# We present the basics of dynamic programming and frame the dynamic pricing as a dynamic programming problem.
#
# In its simplest form, dynamic programming problems consist of two principal features:
#
# - a discrete-time dynamic process
# - and a cost function that is additive over time.
#
# There is a process that happens in stages and an agent is allowed to act in each stage with the objective of minimizing a pre-defined cost function. Every action of the agent must not be considered in isolation. Rather, while deciding what to do in period $k$, the agent should usually weigh in the effects that her current actions might have on subsequent (future) stages. Such a general framework, as one might expect, has applications in very diverse fields such as computer science, operations research and economics.
#
# A dynamic system is usually described as follows:
#
# \begin{equation}
#     x_{k+1} = f_k(x_k, u_k, w_k)
# \end{equation}
#
# where
#
# - $k \in \{0,1...,N-1\}$ is the time index
# - $x_k$ is the state of the system at time $k$
# - $u_k$ is the control variable through which the agent can influence the system's state
# - $w_k$ is the disturbance, a random variable whose realization also determines the system's state
# - $f_k$ is a function that that describes the mechanism through which the system state is updated.
#
#
# To fully characterize the decision problem, a cost function that will steer the agent's decision-making. The total cost is
#
# \begin{equation}
#     g_N(x_N) + \sum\limits_{k=0}^{N-1} g_k(x_k, u_k, w_k)
# \end{equation}
#
# where $g_N(x_N)$ is the final cost occured right at the very end of the process.
# %% [markdown]
# ## Pricing as a dynamic programming problem
#
# - The state $x_k$ is the inventory.
# - The control $u_k$ is the price.  $u_k \in \{5, 10\}$
# - The random noise $w_k$ is the demand.
#
# \begin{equation}
#    x_{k+1} = f(x_k, u_k, w_k) =
#     \begin{cases}
#         x_k - 1 & \text{if $w_k = 1$ }\\
#         x_k & \text{if $w_k = 0$ }\\
#     \end{cases}
# \end{equation}
# %% [markdown]
# ## Cost function
#
# $g_k = $
# %%
import random

import numpy as np

stock = 10  # Initial stock


def pick_strategy():
    if random.random() > 0.5:
        return 5
    return 10


def demand(p):
    mapping = {5: 0.7, 10: 0.3, np.inf: 0}
    try:
        prob = mapping[p]
    except KeyError:
        raise ValueError(f"Price {p} not allowed")

    if random.random() > prob:
        return 0
    return 1


for _ in range(40):
    if stock > 0:
        p = pick_strategy()
    elif stock == 0:
        p = np.inf
    else:
        raise ValueError("Stock cannot be negative")
    q = demand(p)
    stock -= q
    profit = p * q
    print(p, q, profit, stock)

# %%
demand(np.inf)
