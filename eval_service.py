import os
import json
import random
import joblib

import asyncio
import aiohttp
from aiohttp import web
from aiohttp import BodyPartReader

from config import ALGO_PARAMS
from mlrank.submodular.metrics import get_log_likelihood_regularized_score_balanced_components
from mlrank.submodular.optimization.ffs_parallel import eval_new_feature

TEMPORARY_FILE_FOLDER = './temp/'

feature_eval_lock = asyncio.Lock()  # prevent spawning dozens of processes


def estimate_n_jobs(n_cv_ffs: int, decision_function):
    n_cpu = joblib.cpu_count()
    n_jobs = decision_function.n_jobs

    return min(n_cpu / n_jobs, n_cv_ffs)


async def download_huge_file(temp_path: str, field: BodyPartReader) -> dict:
    with open(temp_path, 'wb') as file:
        while True:
            chunk = await field.read_chunk()  # 8192 bytes by default.
            if not chunk:
                break
            file.write(chunk)

    data = joblib.load(temp_path)
    os.remove(temp_path)
    return data


async def collect_params(request: web.BaseRequest) -> dict:
    params = {
        'storage_host': None,
        'storage_port': None,
        'key': None,
        'feature': None,
        'subset': None,
        'decision_function': None,
        'n_cv_ffs': None,
        'ffs_train_share': None,
        'seeds': None,
        'sample': None
    }

    reader = await request.multipart()

    key = random.randint(0, 100000000000)
    f: BodyPartReader = await reader.next()
    while f is not None:
        if f.name == 'sample':
            params[f.name] = await download_huge_file(TEMPORARY_FILE_FOLDER + f'{key}.bin', f)
        elif f.name in params.keys():
            params[f.name] = (await f.read()).decode("utf-8")
        else:
            break

        f: BodyPartReader = await reader.next()

    return params


async def send_to_storage(storage_host: str, storage_port: str, key:str, feature:str, results: dict):
    async with aiohttp.ClientSession() as session:
        data = {
            'result': json.dumps(results),
            'feature': feature,
            'key': key
        }

        response = await session.post(
            f'http://{storage_host}:{storage_port}/store',
            data=data
        )

        if response.status != web.HTTPCreated.status_code:
            raise Exception('results were not sent')

    return True


async def evaluation_worker(params: dict):
    decision_function = None
    for df in ALGO_PARAMS['decision_function']:
        if df['type'] == params['decision_function']:
            decision_function = df['classification']

    n_jobs = estimate_n_jobs(int(params['n_cv_ffs']), decision_function)

    async with feature_eval_lock:
        eval_result = await eval_new_feature(
            subset = params['subset'].split(','),
            new_feature = params['feature'],
            X_f = params['sample']['X_f'],
            X_t = params['sample']['X_t'],
            y = params['sample']['y'],
            n_cv_ffs = int(params['n_cv_ffs']),
            n_jobs = n_jobs,
            seeds = list(map(int, params['seeds'].split(','))),
            train_share = float(params['ffs_train_share']),
            decision_function = decision_function,
            score_function_components = get_log_likelihood_regularized_score_balanced_components,
        )

        return await send_to_storage(
            params['storage_host'],
            params['storage_port'],
            params['key'],
            params['feature'],
            eval_result
        )


async def handle_feature(request: web.Request):
    params = await collect_params(request)
    await evaluation_worker(params)

    raise web.HTTPCreated


app = web.Application()
app.add_routes([web.post('/', handle_feature)])


if __name__ == '__main__':
    os.makedirs(TEMPORARY_FILE_FOLDER, exist_ok=True)
    web.run_app(app, host='0.0.0.0', port=5001)