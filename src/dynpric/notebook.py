import inspect

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


if __name__ == "__main__":
    ...

