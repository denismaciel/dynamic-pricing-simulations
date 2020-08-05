#! /home/denis/.pyenv/versions/dynamic-pricing/bin/python
import sys
import os
from pathlib import Path

from itertools import groupby
from operator import attrgetter

home = Path()

def sync_notebooks():
    targets = (home / "simulations").rglob("nb_*")
    groups = groupby(sorted(targets), lambda x: attrgetter("stem")(x))

    # print("Found notebooks: {}".format(', '.join(str(name) for name, iter_ in groups)))

    for stem, files in groups:
        print(f"Processing {stem}")

        files = list(files)

        if len(files) == 2:
            notebook, py = files
            os.system(f"jupytext --sync {notebook.absolute()}")
        else:
            print(f"Skipping {files}")


def make_markdown(execute=False):
    targets = (home / "simulations").rglob("*.ipynb")
    output_dir = home.absolute() / "docs" / "notebooks"
    notebooks = " ".join([str(nb.absolute()) for nb in targets])
    execute = "--execute" if execute else ""
    cmd = f"""jupyter nbconvert \
            {execute} \
            --ExecutePreprocessor.timeout=180 \
            --output-dir {output_dir} \
            --to markdown {notebooks} \
            """
    print("Running script...")
    print()
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    if sys.argv[1] == "sync_notebooks":
        sync_notebooks()
    elif sys.argv[1] == "make_markdown":
        if sys.argv[2] == "execute":
            make_markdown(execute=True)
        else:
            make_markdown(execute=False)
    else:
        print("Unknown command")

