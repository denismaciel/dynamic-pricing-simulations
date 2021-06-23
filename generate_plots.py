import subprocess

# competition
# demand
# deterministic_models
# exploration_exploitation
# online_network_revenue_management

commands = {
    'replication_online_net_rev_mgmt': [
        'Rscript simulations/replication_online_net_rev_mgmt/graph.R'
    ],
    'exploration_exploitation': [
        'Rscript simulations/exploration_exploitation/bayesian_updating.R',
        'Rscript simulations/exploration_exploitation/graphs.R',
    ],
    'demand': ['python simulations/demand/nb_reservation_price_demand.py'],
    'competition': ['Rscript simulations/competition/analysis.R'],
    'online_network_revenue_management': [
        'python simulations/online_network_revenue_management/analysis.py'
    ],
    'deterministic_models': ['python simulations/deterministic_models/sim.py'],
}


for cmds in commands.values():
    for cmd in cmds:
        print(f'Running `{cmd}`')
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=True,
        )

        if proc.returncode != 0:
            print(f'`{cmd}` failed!')
            exit(1)
