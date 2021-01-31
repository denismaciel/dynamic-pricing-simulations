#! env python
import os
import sys
from itertools import groupby
from operator import attrgetter
from pathlib import Path

root = Path()

# def sync_notebooks():
#     targets = (home / "simulations").rglob("nb_*")
#     groups = groupby(sorted(targets), lambda x: attrgetter("stem")(x))

#     # print("Found notebooks: {}".format(', '.join(str(name) for name, iter_ in groups)))

#     for stem, files in groups:
#         print(f"Processing {stem}")

#         files = list(files)
#         print(files)

#         if len(files) == 2:
#             notebook, py = files
#             # os.system(f"jupytext --sync {notebook.absolute()}")
#         else:
#             print(f"Skipping {files}")

# def make_markdown(execute=False):
#     targets = (home / "simulations").rglob("*.ipynb")
#     output_dir = home.absolute() / "docs" / "notebooks"
#     notebooks = " ".join([str(nb.absolute()) for nb in targets])
#     execute = "--execute" if execute else ""
#     cmd = f"""jupyter nbconvert \
#             {execute} \
#             --ExecutePreprocessor.timeout=180 \
#             --output-dir {output_dir} \
#             --to markdown {notebooks} \
#             """
#     print("Running script...")
#     print()
#     print(cmd)
#     os.system(cmd)

notebooks = [
    "simulations/demand/nb_reservation_price_demand",
    "simulations/dynamic_programming/nb_dynamic_programming",
    "simulations/online_network_revenue_management/simulation",
    "simulations/online_network_revenue_management/analysis",
    "simulations/thompson_sampling/nb_thompson_vs_greedy",
    "simulations/deterministic_models/sim",
]


def make_markdown(path, execute=False):
    notebook_path = Path(path + ".ipynb").absolute()
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook {path} does not exist")
    parent = notebook_path.absolute().parent.stem
    output_dir = root.absolute() / "docs" / "notebooks" / parent
    execute = "--execute" if execute else ""
    cmd = f"""
    jupyter nbconvert \
        {execute} \
        --ExecutePreprocessor.timeout=180 \
        --output-dir {output_dir} \
        --to markdown {notebook_path} \
    """

    # print(cmd)
    os.system(cmd)


for n in notebooks:
    make_markdown(n, execute=True)
