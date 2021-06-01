import multiprocessing
import pathlib
import pickle
import time

import pandas as pd

ROOT_DIR = pathlib.Path()
DATA_DIR = ROOT_DIR / 'data'


def process_log(log, id: int):
    values = []
    for step, record in enumerate(log):
        (value,) = record.values()
        value['id'] = id
        value['step'] = step
        values.append(value)
    return values


def wrangle(path: pathlib.Path) -> None:
    print(f'Processing {path}')
    start = time.time()
    name = path.stem
    with open(path, 'rb') as f:
        results = pickle.load(f)

    logs = [logs for logs, _ in results]
    values = []

    for id, log in enumerate(logs):
        processed = process_log(log, id)
        values.extend(processed)

    df = pd.DataFrame(values)

    df.to_csv(DATA_DIR / f'{name}.csv', index=False)
    end = time.time()

    print(f'Ended {path}. Took {round(end - start)} seconds.')


if __name__ == '__main__':
    # with multiprocessing.Pool(processes=2) as pool:
    #     pool.map(wrangle, DATA_DIR.glob('*.pickle'))
    csv_names = [csv.stem for csv in DATA_DIR.glob('*.csv')]
    pickles = DATA_DIR.glob('*.pickle')
    to_wrangle = [file for file in pickles if file.stem not in csv_names]
    print(to_wrangle)
    for file in to_wrangle:
        wrangle(file)
