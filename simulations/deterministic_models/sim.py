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
# $\pi^*$ can be interpreted as the **marginal opportunity cost of capacity**. For every inventory unit allocated to period $t$, there is an associated opportunity cost of capacity, which is the extra revenue that such a unit would generate if it were allocated to another period $t + x$. Ultimately, for an inventory allocation to be optimal, the marginal revenue must be equal across all periods. Suppose that is not the case for a specific allocation. In that case, the seller can always increase her revenue by moving one unit from a period with lower to a period with higher marginal revenue. It is also interesting to notice that the optimality condition states that the marginal revenue must equal the marginal opportunity cost of capacity. This is equivalent to the previous statement that marginal revenue must be equal across all time periods. The marginal opportunity cost of capacity of period $t$ is given by the marginal revenue the other periods.

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
import warnings
from dynpric.notebook import project_root_dir

FIGS_DIR = project_root_dir() / "figs"
warnings.filterwarnings("ignore")

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
# Let $C = 40$ be the seller's inventory. Now the seller faces a binding constraint and needs to decide how to allocate the 40 units among the two periods. To find the optimal solution for such a simple problem, it is trivial to plug in all possible combinations of stock allocations and check which one yields the highest revenue. As shown in table below, the optimal solution is to allocate 27 units to period 1 (which has less price sensitive customers) and 13 units to period 2. Note that the optimal allocation happens at the point where the marginal revenue of both periods are the closest confirming the analytical solution of the optimization problem above.

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

# %%
# Find q1 that yields highest revenue
boo = df["revenue"] == df["revenue"].max()
q1_max = df[boo]["q1"]

plot = (
    ggplot(df, aes(x="q1", y="revenue"))
    + geom_line()
    + geom_line(aes(y="J1"))
    + geom_line(aes(y="J2"))
    + annotate(
        "text",
        x=(10, 10, 10),
        y=(90, 40, 235),
        label=("Marginal Rev 1", "Marginal Rev 2", "Total Revenue"),
        size=8,
        angle=(-13, 6, 30),
    )
    + geom_vline(xintercept=q1_max, color="red")
    + labs(y="(Marginal) Revenue", x="Inventory allocated to period 1", size=8)
    + theme_light()
    + theme(
        axis_title_x=element_text(size=9),
        axis_title_y=element_text(size=9),
        axis_text_y=element_blank(),
        #         axis_text=element_blank(),
    )
)
plot.save(
    FIGS_DIR / "two_period_marginal_revenue.png", dpi=300, height=3.5, width=4
)
plot

# %% [markdown]
# ## Solving the allocation problem algorithmically
#
# We now simulate a 10-period sales season with linear demands and propose an algorithm to optimally allocate inventory across those periods. The demand function for all periods have the form $q(p) = \alpha - \beta p$. For each period, the intercept $\alpha$ is sampled from the set of integers within the interval $[100, 200]$ with equal probability. Likewise, the slope $\beta$ is sampled from the set of integer $[5, 20]$. Inventory is set to 500 units.
#
# The demand function for each period is represented by `LinearDemand`.

# %%
import random

random.seed(123)

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
        self.q = 0

    def demand(self, p: Price) -> Quantity:
        return self.a - p * self.b

    def inverse_demand(self, q: Quantity = None) -> Price:
        if q is None:
            q = self.q
        return (self.a - q) / self.b

    def marginal_revenue(self, q: Quantity = None) -> Money:
        if q is None:
            q = self.q
        return (self.a - 2 * q) / self.b

    def revenue(self, q: Quantity = None) -> Money:
        if q is None:
            q = self.q
        return q * self.inverse_demand(q)

    def allocate_inventory(self, i: Quantity) -> None:
        self.q += i

    def __repr__(self):
        return (
            f"LinearDemand(a={self.a}, b={self.b}, q={self.q}) "
            f"| Marginal Revenue: {self.marginal_revenue()}"
        )


# %% [markdown]
# Sample demamd functions for the 10 periods.

# %%
def sample_demands(n_periods):
    season = []
    for _ in range(n_periods):
        a = random.randint(100, 200)
        b = random.randint(5, 20)
        demand = LinearDemand(a, b)
        season.append(demand)
    return season


# %% [markdown]
# The plot below depicts the demand functions for all 10 periods. As one can see, there is a great amount of variation across periods, which makes the brute-force approach to finding the optimal allocation less suitable.

# %%
# Generate a demand function for each of the 10 periods
season = sample_demands(10)

plot = ggplot()
for t in season:
    p = np.arange(1, 10)
    q = t.demand(p)
    df = pd.DataFrame({"p": p, "q": q})
    df = df[df["q"] >= 0]
    plot += geom_line(df, aes(x="p", y="q"), alpha=0.5)
plot += labs(x="Price", y="Quantity")
plot += theme_light()
plot += theme(
    axis_title_x=element_text(size=9), axis_title_y=element_text(size=9),
)
plot

# Save plot
plot.save(FIGS_DIR / "deterministic_demands.png", dpi=300, height=3, width=4)

plot

# %% [markdown]
# To allocate the inventory and achieve the highest possible revenue, the seller can follow a relatively simple algorithm.
#
# 1. Find the period with highest marginal revenue.
# 2. If its marginal revenue is greater than zero, allocate one unit of inventory to it. Else, stop the algorithm.
# 3. Check if there is still invenotry left. If yes, return to 1. Else, stop the algorithm.

# %%
season = sample_demands(10)


def allocate_inventory(season, inventory: int):
    remaining = inventory

    for _ in range(inventory):
        marginal_revenues = [d.marginal_revenue() for d in season]
        max_ = max(marginal_revenues)

        if max_ > 0:
            idx = marginal_revenues.index(max_)
            season[idx].allocate_inventory(1)
            remaining -= 1
        else:
            break
    return season, remaining


INVENTORY = 500
season, remaining = allocate_inventory(season, INVENTORY)

# %%
print(remaining)

# %%
final_allocation = pd.DataFrame(
    [
        {
            "Period": t,
            "Alpha": d.a,
            "Beta": d.b,
            "Marginal Revenue": round(d.marginal_revenue(), 2),
            "Allocated Quantity": d.q,
        }
        for t, d in enumerate(season)
    ]
)
final_allocation

# %%
plot = ggplot()
q = np.arange(1, 200)

for d in season:
    revenue = d.revenue(q)
    df = pd.DataFrame({"q": q, "revenue": revenue})
    df = df[df["revenue"] > 0]
    plot += geom_line(df, aes(x="q", y="revenue"), alpha=0.5)

    allocated_q = pd.DataFrame({"q": [d.q], "revenue": [d.revenue()]})
    plot += geom_point(allocated_q, aes(x="q", y="revenue"))

plot += labs(x="Quantity", y="Revenue")
plot += theme_light()

plot.save(FIGS_DIR / "deterministic_revenue.png", dpi=300, height=3, width=4)
plot
