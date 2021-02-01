# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
# %%
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# %%
import numpy as np
import pandas as pd
from plotnine import *
import matplotlib.pyplot as plt

# %%
theme_set(theme_light())
FIGS_DIR = Path(os.environ["FIGS_DIR"])


# %%
def simulate_wip():
    λ = 100

    θ_lo = 0.6
    θ_hi = 0.4

    β_lo = 10
    β_hi = 3 * β_lo

    n = np.random.poisson(λ, size=1)
    n_lo, n_hi = np.random.multinomial(n, (θ_lo, θ_hi))

    wips = pd.concat(
        [
            pd.DataFrame(
                {"wip": np.random.exponential(β_lo, n_lo), "group": "low"}
            ),
            pd.DataFrame(
                {"wip": np.random.exponential(β_hi, n_hi), "group": "high"}
            ),
        ]
    )

    return wips


wips = [simulate_wip() for _ in range(50)]


# %%
def demand_fn(wip, prices):
    """Calculates the demand for every price point in `prices`
    """
    # Demand q for price p
    q = lambda wip, p: (wip["wip"] > p).sum()
    pq_pairs = [(p, q(wip, p)) for p in prices]
    return pd.DataFrame(pq_pairs, columns=["p", "q"])


curves = [demand_fn(wip, range(0, 100, 2)) for wip in wips]

# %%
demand_curve = ggplot()

for curve in curves:
    demand_curve += geom_line(curve, aes("p", "q"), alpha=0.1)

demand_curve = (
    demand_curve
    + labs(y="Quantity q", x="Price p")
    + theme(text=element_text(size=8))
)

demand_curve.save(FIGS_DIR / "wip_demand.png", dpi=300, height=3, width=3)
demand_curve

# %%
all_wips = pd.concat(wips)
pt99 = np.percentile(all_wips["wip"], 99)

wip_distribution = (
    ggplot(pd.concat(wips), aes(x="wip", fill="group"))
    + geom_histogram(position="identity", bins=80, alpha=0.5)
    + labs(y="Count", x="Willigness to Pay", fill="")
    + xlim((0, pt99))
    + theme(text=element_text(size=8), legend_position="none")
)
wip_distribution.save(
    FIGS_DIR / "wip_distribution.png", dpi=300, height=3, width=3
)
wip_distribution
