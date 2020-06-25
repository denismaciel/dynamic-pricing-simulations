#! /home/denis/.pyenv/versions/dynamic-pricing/bin/python
import sys
import os
from pathlib import Path

from itertools import groupby
from operator import attrgetter

home = Path()

def sync_notebooks():
    targets = (home / "src").rglob("nb_*")
    groups = groupby(sorted(targets), lambda x: attrgetter("stem")(x))

    for stem, files in groups:
        print(f"Processing {stem}")

        files = list(files)

        if len(files) == 2:
            notebook, py = files
        else:
            print(f"Skipping {files}")

        os.system(f"jupytext --sync {notebook.absolute()}")

def make_markdown():

    targets = (home / "src").rglob("*.ipynb")

    output_dir = home.absolute() / "docs" / "notebooks"
    notebooks = " ".join([str(nb.absolute()) for nb in targets])

    cmd = f"""jupyter nbconvert --output-dir {output_dir} --to markdown {notebooks}"""
    print("Running script...")
    print()
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    if sys.argv[1] == "sync_notebooks":
        sync_notebooks()
    elif sys.argv[1] == "make_markdown":
        make_markdown()
    else:
        print("Unknown command")

