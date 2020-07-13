from scipy.optimize import linprog

prices = (29.9, 34.9, 39.9, 44.9)
demand = (0.8, 0.6, 0.3, 0.1)
assert len(prices) == len(demand)

alpha = 0.5
T = 10_000
inventory = alpha * T
c = inventory / T

# The reason for the minus sign is that scipy only does minimizations
objective = [-(p * d) for p, d in zip(prices, demand)]

# --- Constraints ---:

# 1. Demand is smaller equal than available inventory
c1 = [demand, c]

# Sum of probabilities smaller equal zero
c2 = [(1, 1, 1, 1,), 1]

# 3. Probability of picking a price must be greater than zero
c3 = [(-1, -1, -1, -1,), 0]

constraints = [c1, c2, c3]

lhs_ineq = []
rhs_ineq = []

for lhs, rhs in constraints:
    lhs_ineq.append(lhs)
    rhs_ineq.append(rhs)


opt = linprog(c=objective, A_ub=lhs_ineq, b_ub=rhs_ineq, method='revised simplex')

