import multiprocessing
import pathlib
import pickle
import time

import pandas as pd


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

    df.to_csv(f'data/{name}.csv', index=False)
    end = time.time()

    print(f'Ended {path}. Took {round(end - start)} seconds.')


if __name__ == '__main__':
    # SIMULATION_NAME = 'ts_fixed_with_bernoulli'
    # SIMULATION_NAME = 'ts_update_with_bernoulli'
    # SIMULATION_NAME = 'ts_ignore_inventory_with_bernoulli'

    with multiprocessing.Pool(processes=2) as pool:
        pool.map(wrangle, pathlib.Path('data').glob('*.pickle'))

    # for path in pathlib.Path('data').glob('*.pickle'):
    #     wrangle(path)
