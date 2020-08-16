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

# %%
import warnings

from plotnine import *
import pandas as pd

from dynpric.notebook import project_root_dir

warnings.filterwarnings('ignore')

N_TRIALS = 500
N_PERIODS = 500
FIGS_DIR = project_root_dir() / "figs"


# Read in simulation data
ts_fixed = pd.read_parquet(f"data/ts_fixed_trials{N_TRIALS}_periods{N_PERIODS}.parquet")
clairvoyant = pd.read_parquet(
    f"data/clairvoyant_trials{N_TRIALS}_periods{N_PERIODS}.parquet"
)

# %%
ts_fixed.period_revenue.mean(), clairvoyant_avg_revenue

# %%
revenue = ts_fixed.groupby("t").period_revenue.mean()
clairvoyant_avg_revenue = clairvoyant.period_revenue.mean()

revenue_over_time = (
    ggplot(aes(revenue.index, revenue))
    + geom_line()
    + lims(y=(0, 15))
    + geom_hline(aes(yintercept=clairvoyant_avg_revenue), color="red")
    + labs(y="Revenue", x='Periods')
)

revenue_over_time.save(FIGS_DIR / 'online_net_reveue_over_time.png', dpi=300, height=3, width=3)
revenue_over_time

# %%
df = (
    ts_fixed[["t", "price_29.9", "price_34.9", "price_39.9", "price_44.9"]]
    .groupby("t")
    .mean()
    .reset_index()
    .melt(id_vars="t", var_name="price", value_name="pp")
)
df

# %%
(ggplot(df, aes(x="t", y="pp", color="price")) + geom_line() + facet_wrap("price"))

# %%
counts_per_step = ts_fixed.groupby(["t", "price"]).size().reset_index(name="n")
counts_per_step["pp"] = counts_per_step["n"] / N_TRIALS

plot = (
    ggplot(counts_per_step, aes("t", "pp", color="factor(price)"))
    + geom_line()
    + labs(
        title="How often price x was offered in period t averaged across all trials",
        y="%",
        color="Price Levels",
    )
    + lims(y=(0, 1))
    + facet_wrap("price")
)

plot

# %%
counts_per_step.groupby(["price"]).pp.mean()

# %%
pd.DataFrame(
    {
        "clairvoyant": clairvoyant.groupby("t").period_revenue.mean(),
        "ts_fixed": ts_fixed.groupby("t").period_revenue.mean(),
    }
)
