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
# # Single product with limited inventory
#
# Fashion retail and industries that sell and produce seasonal goods often face the challenge of pricing their products without the possibility of reordering (or resuming production) in case they run out of inventory during the selling season. This is usually the case because the sales season tends to be much shorter than production and ordering cycles. When choosing the price path of such products, the seller needs to consider the constraint that the total sales shall not exceed the initial inventory available at the beginning of the sales season.
#
# A fundamental way of modeling this problem is by considering the product in isolation and assuming the customers to be myopic. Demand can then be modeled as a function of the price at the current period only. Since the customers are myopic, they do not consider that the price might change in the future, making it eventually more attractive for them to delay the purchase. That is, customers do not act strategically. Faced with a price today, they make a purchase decision based solely on the current price and do not consider how it might develop in the future.
#
# Moreover, in this setting, we consider only one product. We do not model the more realistic scenario in which the prices of other related goods determine a product's demand. We assume there are no substitutes or compliments capable of affecting demand for the product under consideration.
#
# The assumptions result in a demand function that depend on the price $p(t)$ at current time $t$ and on $t$ itself. We denote demand at time $t$ by $d(t, p(t))$. We also pose some regularity properties on the demand function:
#
# - The demand is continuously differentiable and strictly decreasing on price $d'(t, p) < 0$, which implies it has an inverse $p(t, d)$.
# - For sufficiently high prices, demand is zero $\inf_p d(t, p) = 0$
# - The revenue function $r(t,p) = pd(t,p) = p(t,d)d$ is finite.
# - The marginal revenue J as a function of demand is strictly decreasing in $d$
#
# \begin{equation}
#     J(t, d) \equiv \frac{\partial}{\partial d} r(t,d) = p(t,d) + dp'(t,d) < 0
# \end{equation}
#
# ## Deterministic Models
#
# Deterministic models are models that do not entail randomness. Given a price p, demand is certain to be the deterministic quantity $q = f(p)$.
#
# A simple deterministic dynamic pricing model is one where, although allowed to chage over time, the demand function is known at each time $t$ and there is no random factor in it. That is, given we $p(t)$ in time $5$, we know the exact value $d(t, p(t))$ that will be demanded.
#
# For the seller, the maximization problem boils down to achieving the highest revenue while respecting the constraint that total demand across all periods shall not exceed the initial inventory.
#
# \begin{equation}
#    \max \sum^{T}_{t=1} = r(t,d(t)) \\
#    \text{s.t.} \quad \sum^{T}_{t=1}d(t) \le C \\
#    \quad d(t) \ge 0
# \end{equation}
#
# Solving the optimization problem, we have that
#
# \begin{equation}
#     J(t, d^*(t)) = \pi^*
# \end{equation}

# %% [markdown]
# where $\pi^*$ is the Lagrange multiplier on the inventory constraint.
#
# $\pi^*$ can be interpreted as the **marginal opportunity cost of capacity**. For every inventory unit allocated to period $t$, there is an associated opportunity cost of capacity, which is the extra revenue that such a unit would generate if it were allocated to another period $t + x$. Ultimately, for an inventory allocation to be optimial, the marginal revenue must be equal across all periods. If that is not the case for a specific allocation, the seller can always increase her revenue by moving one unit from a period with lower to a period with higher marginal revenue. It is also interesting to notice that the optimality condition states that the marginal revenue must equal the marginal opportunity cost of capacity. This is equivalent to the previous statemet that marginal revenue must be equal across all time periods. The marginal opportunity cost of capacity of period $t$ is given by the marginal revenue the other periods.

# %% [markdown]
# ## A numerical example
#
# We illustrate the deterministic framework explained above with a simple two-period sellling season. Demand is given by $d_1 = -p_1 + 100$ in the first period and $d_2 = -2p_2 + 120$ in the second period. If we were to consider the case with unlimited inventory, each period maximization problem could be solved independently and we would have the following solution:
#
# \begin{equation}
# r_1 = p_1(-p_1 + 100)   \rightarrow p_1^* = 50, q_1^* = 50   \\
# r_2 = p_2 (-2p_2 + 120) \rightarrow p_2^* = 30, q_2^* = 60
# \end{equation}

# %%
import numpy as np
import pandas as pd
from plotnine import *

# Demands
d1 = lambda p: -p + 100
d2 = lambda p: -2 * p + 120

# Inverse Demands
p1 = lambda q: -q + 100
p2 = lambda q: -q / 2 + 120 / 2

# Revenue
r1 = lambda p: p * d1(p)
r2 = lambda p: p * d2(p)

# Marginal Revenue
J1 = lambda q: -2 * q + 100
J2 = lambda q: -q + 60

# %% [markdown]
# Let $C = 40$ be the seller's inventory. Now the seller faces a binding constraint and needs to decide how to allocate the 40 units among the two periods. For such a simple problem, we just need to plug in all possible combinations of stock allocations and check which one yields the highest overall revenue. The optimal solution is to allocate 27 units to period 1 (which have less price sensitive customers) and 13 units to period 2. Note that the optimal allocation happens at the point where the marginal revenue of both periods are the closest.

# %%
C = 40

df = pd.DataFrame({"q1": np.arange(C + 1)})
df["q2"] = C - df["q1"]
df["revenue"] = df.apply(
    lambda row: r1(p1(row["q1"])) + r2(p2(row["q2"])), axis=1
)
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
    + labs(y="(Marginal) Revenue", x="Inventory Offered in Period 1")
    + theme_minimal()
)


# %%
def demand_factory(n: int):

    # Demand form is a + b * p
    # Derivative of the revenue function r = p (a + b * p)
    # will be J = a + 2*b*p

    # Inverse demand form  p(q) = a - bq
    # Revenue function     r(q) = q*p(q) = aq - bq^2
    # Marginal revenue     J(q) = a - 2qb

    for _ in range(n):
        print(_)

demand_factory(5)

# %%
import random

Price = float
Quantity = int
Money = float

class LinearDemand:
    """
    Demand function of the form: q(p) = a - b*p
    
        Inverse demand:   p(q)  = (a - q)/b
        Marginal revenue: mr(q) = (a - 2q)/b
        Revenue:          r(q)  = p(q)*(q)
    """
    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b
        
    def demand(self, q):
        return self.a - q * self.b
    
    def inverse_demand(self, q: Quantity) -> Price:
        return (self.a - q) / self.b

    def marginal_revenue(self, q: Quantity) -> Money:
        return (self.a - 2*q) / self.b

    def revenue(self, q: Quantity) -> Money:
        return q * self.inverse_demand(q)
    
    def __repr__(self):
        return f"LinearDemand({self.a}, {self.b})"


# def marginal_revenue_factory(a, b):
#     def marginal_revenue(q):
#         return a - 2 * q * b
#     return marginal_revenue


season = []
for _ in range(10):
    a = random.randint(100, 200)
    b = random.randint(5, 20)
    demand = LinearDemand(a, b)
    season.append(demand)
    
plot = ggplot()
for t in season:
    p = np.arange(1, 10)
    q = t.demand(np.arange(1, 10))
    df = pd.DataFrame({'p': p, 'q': q})
    plot += geom_line(df, aes(x = 'p', y = 'q'))

# %%
plot

# %%
