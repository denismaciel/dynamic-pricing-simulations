# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupyteft:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # Analysis of simulation results
#
# In the following, we analyze the simulation results of **Ferreira, Kris Johnson, David Simchi-Levi, and He Wang. “Online Network Revenue Management Using Thompson Sampling”**.
# %%
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import *

warnings.filterwarnings('ignore')

N_TRIALS = 500
N_PERIODS = 500
FIGS_DIR = Path(os.environ['FIGS_DIR'])

# Read in simulation data
ts_fixed = pd.read_parquet(
    f'data/ts_fixed_trials{N_TRIALS}_periods{N_PERIODS}.parquet'
)
clairvoyant = pd.read_parquet(
    f'data/clairvoyant_trials{N_TRIALS}_periods{N_PERIODS}.parquet'
)

# %% [markdown]
# At the end of each period, we recorded metrics about the state of the trial. The information gathered was:
#
# - price: the price set by the seller
# - period_revenue: the revenue collected
# - belief_x : the mean value of the distribution of the demand parameter the seller believes is associated with price x
#
#  The resulting dataframe has a row for each period (`t`) of every trial (`s`).

# %% [markdown]
# ## Revenue
#
# We look at the revenue achieved by the seller at period t averaged across all trials. The red straight line shows the average revenue achieved by a clairvoyant seller, whereas the wiggly black line shows the average revenue for each period achieved by TS-fixed.
#
# In the first periods, the seller makes around half the revenue that a clairvoyant would make. It takes her around 150 periods (roughly 30% of the selling season) to achieve the revenue level of the clairvoyant. From then on, the revenue stabilizes and fluctuates around the 10 dollar mark.

# %%
revenue = ts_fixed.groupby('t').period_revenue.mean()
clairvoyant_avg_revenue = clairvoyant.period_revenue.mean()

revenue_over_time = (
    ggplot(aes(revenue.index, revenue))
    + geom_line()
    + lims(y=(0, 15))
    + geom_hline(aes(yintercept=clairvoyant_avg_revenue), color='red')
    + labs(y='Revenue', x='Periods')
)

revenue_over_time.save(
    FIGS_DIR / 'online_net_revenue_over_time.png', dpi=300, height=3, width=5
)
revenue_over_time

# %% [markdown]
# ## Offered prices
#
# It is also interesting to analyze how often a price was offered at each period t. 44.9 dominates the very first periods, because of the prior beliefs the seller has. All prices have the same $Beta(1,1)$ prior. The seller believes that the demand parameter can be any number between 0 and 1 with equal probability. Given such state of affairs, it's no wonder that 44.9 yields the highest revenue the vast majority of times after pinning down a value for the demand parameters via Thompson sampling.
#
# Remeber that the optimal price strategy if the seller were clairvoyant is to offer 39.9 with probability 0.75 and 44.9 with probability 0.25. The TS-fixed strategy approaches the clairvoyant's as the selling season unravels. Around period 300, the average of TS-fixed is virtually the same as that of a clairvoyant seller.

# %%
counts_per_step = ts_fixed.groupby(['t', 'price']).size().reset_index(name='n')
counts_per_step['pp'] = counts_per_step['n'] / N_TRIALS

pricing_strategy = (
    ggplot(counts_per_step, aes('t', 'pp', color='factor(price)'))
    + geom_line()
    + labs(
        title='How often price x was offered in period t averaged across all trials',
        y='%',
        color='Price Levels',
    )
    + lims(y=(0, 1))
    + facet_wrap('price')
    + theme(legend_position='none')
)

pricing_strategy.save(
    FIGS_DIR / 'online_net_pricing_strategy_over_time.png',
    dpi=300,
    height=3.3,
    width=5,
)

pricing_strategy

# %% [markdown]
# ## The learning process
#
# Finally, we can look into how the seller processes new information. In this model, new information consit of the pair price set and realized demand. As mentioned already, every price has a prior distribution $Beta(1,1)$. For every period where a price is chosen and given that the posterior is currently $Beta(\alpha, \beta)$, its posterior is updated as follows:
#
# - if demand is 1, the posterior is updated to $Beta(\alpha + 1, \beta)$
# - if demand is 0, the posterior is updated to $Beta(\alpha, \beta + 1)$
#
# The plot shows the average expected value from the posterior distribution of the demand parameter over time for each available price. For the very first period, the expected value averaged across simulations is close to 0.5 for all prices but 44.9. Why is that the case? Because 44.9 is by far the most often chosen price in the first period and for that reason is usually the first price level to have its beliefs updated. When the price is 44.9, demand will be 1 with only 0.1 probability. The posterior after the first period will then be be $Beta(1, 2)$ 90\% of the time and $Beta(1, 2)$ the other 10\%. In fact, it's staright forward to calculate the expected value of the mean of the posterior distribution of 44.9 right after the first period.
#
# Knowing that the expected value of $X \sim Beta(\alpha, \beta)$ is $ E[X]= \frac{\alpha}{\alpha + \beta}$ and that $q | p = 44.5 \sim Beta(1, 1)$ in the first period, we have
#
# \begin{equation}
#     E[q | p = 44.5] = 0.9 \frac{1}{3} + 0.1 \frac{2}{3} = 0.3\bar{6}
# \end{equation}
#
# which is roughly the starting point of the line 44.9 in the plot.
#
# It is also worth noting that the seller learns the true demands of 39.9 and 44.9 quite accurately. She also gets close to the true demand of 34.9, but completely misses the true demand of 29.9. This is actually a feature of Thompson sampling. It tends to learn more about the possibilities that seem most promising with the information available in the present.

# %%
df = (
    ts_fixed[['t', 'belief_29.9', 'belief_34.9', 'belief_39.9', 'belief_44.9']]
    .groupby('t')
    .mean()
    .reset_index()
    .melt(id_vars='t', var_name='price', value_name='pp')
)

belief_development = (
    ggplot(df, aes(x='t', y='pp', color='price'))
    + geom_line()
    + labs(y='Probability')
    + lims(y=(0, 1))
    + facet_wrap('price')
    + theme(legend_position='none')
)

belief_development.save(
    FIGS_DIR / 'online_net_beliefs_over_time.png', dpi=300, height=3.5, width=5
)
belief_development

# %%
df.query('t == 499')
