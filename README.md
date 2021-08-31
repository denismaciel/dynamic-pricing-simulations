# Dynamic Pricing Simulations

This repository contains the code used to create the simulations and plots in
my master thesis "Demand Learning under Limited Inventory: a simulation-based
study."

## Requirements

To reproduce the results, you must have Python 3.9 or greater and R 4.1.1 or greater installed on your system.

### Setting up Python

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

### Setting up R

```bash
R -e 'install.packages("tidyverse")'
```

## Running the simulations and generating the plots

Before start running the code, make sure you have the virtual environment activated and environment variables set by running:

```bash
export FIGS_DIR=/home/denis/Personal/MasterThesis/dynamic-pricing-simulations/figs
source venv/bin/activate
```

### `deterministic_models`

To reproduce Table 1, Table 2 and Figure 1:

```bash
python3 simulations/deterministic_models/sim.py
```



### `exploration_exploitation`

To reproduce Figure 2 and Figure 3:

```bash
Rscript simulations/exploration_exploitation/bayesian_updating.R
```

To reproduce Figure 4 and Figure 5:

```bash
# Run simulations
python3 simulations/exploration_exploitation/sim.py simulate

# Transform pickle into CSV to be consumed by R.
python3 simulations/exploration_exploitation/sim.py generate_csv

# Generate plots
Rscript simulations/exploration_exploitation/graphs.R
```

### `online_network_revenue_management`

To reproduce Figure 6, Figure 7 and Figure 8:

```bash
# Run simulations
python3 simulations/online_network_revenue_management/simulation.py

# Generate plots
python3 simulations/online_network_revenue_management/analysis.py
```

### `replication_online_net_rev_mgmt`

To reproduce For Figure 9:

```bash
# Run simulation with different sales season lengths
python3 simulations/replication_online_net_rev_mgmt/sim.py simulate --n-periods 100,500,1000

# Generate CSVs
python3 simulations/replication_online_net_rev_mgmt/wrangle.py

# Generate plot
Rscript simulations/replication_online_net_rev_mgmt/graph.R
```

### `demand`

To reproduce Figure 11:

```bash
python3 simulations/demand/nb_reservation_price_demand.py
```

### `competition`

To reproduce Figure 12 and Figure 13:

```bash
# Run simulations
python3 -m simulations.competition.sim simulate

# Generate CSVs
python3 -m simulations.competition.sim generate_csv

# Generate plots
Rscript simulations/competition/analysis.R
```
