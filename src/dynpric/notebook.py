import inspect
from pathlib import Path

from IPython.display import display, Markdown


def display_source_code(x, method_name=None):
    def display_code_block(x):
        code_block = "\n".join(["```python", f"{inspect.getsource(x)}", "```"])
        display(Markdown(code_block))

    if method_name is not None:
        members = inspect.getmembers(x)

        method = [content for name, content in members if name == method_name]
        assert len(method) == 1
        method = method[0]
        display_code_block(method)
        return

    display_code_block(x)


def project_root_dir():
    """
    The project root directory is considered the first 
    directory with .git folder in it
    """
    p_origin = p = Path().absolute()
    while True:
        if p == Path("/"):
            raise FileNotFoundError(f"Could not find project root dir from {p_origin}")
        if p / Path(".git") in p.iterdir():
            return p.absolute()
        p = p.parent.absolute()


if __name__ == "__main__":
    print(project_root_dir())

