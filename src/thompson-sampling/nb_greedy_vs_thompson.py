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
# Every arm starts with equal probability
# Pick the arm with highest mean (if means are equal, pick randomly)
# Update mean with results
# Rinse and repeat

from typing import NamedTuple
import random
import matplotlib.pyplot as plt


def roll_for_best(idx: int) -> int:
    probs = [0.9, 0.8, 0.7]
    prob = probs[idx]
    if random.random() > prob:
        return 0
    return 1


def simulate(pick_best, params, update_params):
    WORLDS = []

    for _ in range(10000):
        T = []
        to_update_params = params
        for t in range(1000):
            best_idx = pick_best(to_update_params)
            result = roll_for_best(best_idx)
            to_update_params = update_params(to_update_params, best_idx, result,)
            T.append((t, best_idx))
        WORLDS.append(T)

    d = {
        0: [[] for _ in range(1000)],
        1: [[] for _ in range(1000)],
        2: [[] for _ in range(1000)],
    }

    for w in WORLDS:
        for i, t in enumerate(w):
            _, best_idx = t
            d[0][i].append(best_idx == 0)
            d[1][i].append(best_idx == 1)
            d[2][i].append(best_idx == 2)

    return d


def plot(d):
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(range(1000), [sum(x) / len(x) for x in d[2]])
    ax.plot(range(1000), [sum(x) / len(x) for x in d[1]])
    ax.plot(range(1000), [sum(x) / len(x) for x in d[0]])


# %%
def update_means(params, idx, result):
    p = params[idx]
    mean = (p.mean * p.n_realizations + result) / (p.n_realizations + 1)
    n_realizations = p.n_realizations + 1

    temp = list(params)
    temp[idx] = MeanParam(mean, n_realizations)
    return tuple(temp)


def pick_best_mean(params) -> int:
    means = [x.mean for x in params]
    maxi = max(means)
    maxi_idxs = [i for i, mean in enumerate(means) if mean == maxi]

    if len(maxi_idxs) == 1:
        return maxi_idxs[0]
    else:
        return random.choice(maxi_idxs)


class MeanParam(NamedTuple):
    mean: int
    n_realizations: int


params_test = (MeanParam(0, 0), MeanParam(7, 2))
r = update_means(params_test, 1, 1)
assert r[1].mean == 5
assert r[1].n_realizations == 3
assert pick_best_mean(params_test) == 1


mean_params = (
    MeanParam(0, 0),
    MeanParam(0, 0),
    MeanParam(0, 0),
)

d = simulate(pick_best_mean, mean_params, update_means)
plot(d)


# %%
def update_thom(params, idx, result):
    p = params[idx]

    if result not in (0, 1):
        raise ValueError("`result` must be either 0 or 1")
    
    a = p.a + result
    b = p.b + (1 - result)

    temp = list(params)
    temp[idx] = ThomParam(a, b)
    return tuple(temp)

def pick_best_thom(params):
    from numpy.random import beta
    
    samples = []
    for p in params:
        samples.append(beta(p.a, p.b))
    return samples.index(max(samples))

class ThomParam(NamedTuple):
    a: int
    b: int
        
thom_params = (
    ThomParam(1, 1),
    ThomParam(1, 1),
    ThomParam(1, 1),
)

d = simulate(pick_best_thom, thom_params, update_thom)
plot(d)

# %%
[sum(x) / len(x) for x in d[0]][999], [sum(x) / len(x) for x in d[1]][999], [sum(x) / len(x) for x in d[2]][999]


# %%
