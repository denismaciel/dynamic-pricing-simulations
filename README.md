# Dynamic Pricing Simulations

Simulation-based comparison between dynamic pricing algorithms

## How to render notebooks

Make sure all relevant notebooks are listed in `PATH` in`scripts/compile_notebooks.py`.

Run:

```
FIGS_DIR=/home/denis/Personal/MasterThesis/dynamic-pricing-simulations/figs ./venv/bin/python3 scripts/compile_notebooks.py
```
This will sync `.py` with `.ipynb` and render `.ipynb` as `.md`. All notebooks
will have their content executed. To change that (why would you? Doesn't hurt to
wait a little), toggle the `execute` variable in `main` function by hand.
