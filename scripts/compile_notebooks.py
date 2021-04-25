#! env python
import shutil
import subprocess
from pathlib import Path

ROOT = Path()

notebooks = [
    'simulations/demand/nb_reservation_price_demand',
    'simulations/dynamic_programming/nb_dynamic_programming',
    'simulations/online_network_revenue_management/simulation',
    'simulations/online_network_revenue_management/analysis',
    'simulations/thompson_sampling/nb_thompson_vs_greedy',
    'simulations/deterministic_models/sim',
]


def center_text(text: str, sep='=') -> str:
    width, height = shutil.get_terminal_size()
    return f'   {text}   '.center(width, sep)


def nb_from_py(path: str) -> str:
    _path = Path(f'{path}.py')

    if not _path.exists():
        raise FileNotFoundError(f'Python file does not exist: {path}')

    proc = subprocess.run(
        f'jupytext --to ipynb {_path}',
        capture_output=True,
        text=True,
        shell=True,
    )

    if proc.returncode != 0:
        raise Exception(proc.stderr)

    return proc.stdout


def make_markdown(path, *, execute=False):
    notebook_path = Path(path + '.ipynb').absolute()
    if not notebook_path.exists():
        raise FileNotFoundError(f'Notebook {path} does not exist')
    parent = notebook_path.absolute().parent.stem
    output_dir = ROOT.absolute() / 'docs' / 'notebooks' / parent
    execute = '--execute' if execute else ''
    cmd = ' '.join(
        (
            'jupyter nbconvert',
            f'{execute}',
            '--ExecutePreprocessor.timeout=180',
            f'--output-dir {output_dir}',
            f'--to markdown {notebook_path}',
        )
    )
    print('\t Command: ', cmd)

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        shell=True,
    )

    if proc.returncode != 0:
        raise Exception(proc.stderr)

    return proc.stdout


def main() -> int:

    for path in notebooks:
        print(center_text(f'Processing {path}'))
        print('\t py ===> ipynb')
        _ = nb_from_py(path)
        print('\t ipynb ===> md')
        make_markdown(path, execute=True)
        print('\n\n')

    return 0


if __name__ == '__main__':
    main()
