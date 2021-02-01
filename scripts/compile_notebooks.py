#! env python
import shutil
import subprocess
import sys
from itertools import groupby
from operator import attrgetter
from pathlib import Path

ROOT = Path()

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


def center_text(text: str, sep="=") -> str:
    width, height = shutil.get_terminal_size()
    return f"   {text}   ".center(width, sep)


def nb_from_py(path: str) -> str:
    path = Path(f"{path}.py")

    if not path.exists():
        raise FileNotFoundError(f"Python file does not exist: {path}")

    proc = subprocess.run(
        f"jupytext --to ipynb {path}",
        capture_output=True,
        text=True,
        shell=True,
    )

    if proc.returncode != 0:
        raise Exception(proc.sterr)

    return proc.stdout


def make_markdown(path, *, execute=False):
    notebook_path = Path(path + ".ipynb").absolute()
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook {path} does not exist")
    parent = notebook_path.absolute().parent.stem
    output_dir = ROOT.absolute() / "docs" / "notebooks" / parent
    execute = "--execute" if execute else ""
    cmd = f"""
    jupyter nbconvert \
        {execute} \
        --ExecutePreprocessor.timeout=180 \
        --output-dir {output_dir} \
        --to markdown {notebook_path} \
    """

    proc = subprocess.run(cmd, capture_output=True, text=True, shell=True,)

    if proc.returncode != 0:
        raise Exception(proc.stderr)

    return proc.stdout


def main() -> int:

    for path in notebooks:
        print(center_text(f"Processing {path}"))
        print(center_text("py ===> ipynb", sep="-"))
        result = nb_from_py(path)
        print(center_text("ipynb ===> md", sep="-"))
        make_markdown(path, execute=True)

    return 0


# for n in notebooks:
#     make_markdown(n, execute=True)


if __name__ == "__main__":
    main()
