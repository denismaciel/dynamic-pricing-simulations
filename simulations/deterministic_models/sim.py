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
# # Deterministic Moldels
#

# %%
import numpy as np
import pandas as pd
from plotnine import *

# %%
d1 = lambda p: -p + 100
d2 = lambda p: -2 * p + 120

p1 = lambda q: -q + 100
p2 = lambda q: -q / 2 + 120 / 2

r1 = lambda p: p * d1(p)
r2 = lambda p: p * d2(p)

# Derivatives
J1 = lambda q: -2*q + 100
J2 = lambda q: -q + 60

# %%
df = pd.DataFrame({"q1": np.arange(C + 1)})
df["q2"] = C - df["q1"]
df["revenue"] = df.apply(lambda row: r1(p1(row["q1"])) + r2(p2(row["q2"])), axis=1)
df["J1"] = J1(df["q1"])
df["J2"] = J2(df["q2"])

df[20:30]

# %%
# Scale down the revenue for better visualizations
df["revenue"] = df["revenue"] / 10
df["J1"] = df["J1"]
df["J2"] = df["J2"]


boo = df["revenue"] == df["revenue"].max()
q1_max = df[boo]["q1"]

(
    ggplot(df, aes(x="q1", y="revenue"))
    + geom_line()
    + geom_line(aes(y="J1"))
    + geom_line(aes(y="J2"))
    + annotate(
        "text",
        x=(10, 10, 10),
        y=(90, 40, 235),
        label=("Marginal Rev 1", "Marginal Rev 2", "Total Revenue"),
        size=10,
        angle=(-13, 6, 30),
    )
    + geom_vline(xintercept=q1_max, color="red")
    + labs(y="(Marginal) Revenue", x="Demand in Period 1")
    + theme_minimal()
)


# %%
def demand_factory(n: int):
    
    # Demand form is a + b * p
    # Derivative of the revenue function r = p (a + b * p)
    # will be J = a + 2*b*p
    
    # Inverse demand form  p(q) = b - aq 
    # Revenue function     r(q) = q*p(q) = bq - aq^2
    # Marginal revenue     J(q) = b - 2qavim
    
    for _ in range(n):
        print(_)
    
demand_factory(5)
