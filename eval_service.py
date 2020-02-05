import os
import json
import random
import joblib

import asyncio
from aiohttp import web
from aiohttp import BodyPartReader

from config import ALGO_PARAMS

from mlrank.submodular.metrics import get_log_likelihood_regularized_score_balanced_components
from mlrank.submodular.optimization.ffs_parallel import eval_new_feature

TEMPORARY_FILE_FOLDER = './temp/'
lock = asyncio.Lock()
file_lock = asyncio.Lock()


def estimate_n_jobs(n_cv_ffs: int, decision_function):
    n_cpu = joblib.cpu_count()
    if hasattr(decision_function, 'n_jobs') and decision_function.n_jobs is not None:
        n_jobs = decision_function.n_jobs
    else:
        n_jobs = 1

    return int(min(n_cpu / n_jobs, n_cv_ffs))


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

    for k, v in params.items():
        if v is None:
            raise Exception(f"{k} must be set")

    return params


async def evaluation_worker(params: dict):
    decision_function = None
    for df in ALGO_PARAMS['decision_function']:
        if df['type'] == params['decision_function']:
            decision_function = df['classification']

    n_jobs = estimate_n_jobs(int(params['n_cv_ffs']), decision_function)
    subset = params['subset'].split(',')
    if subset[0] == '':
        subset = []

    async with lock:
        await asyncio.sleep(1)
        eval_result = {'ll': 12, 'llcf': 50}
        #eval_result = eval_new_feature(
        #    subset = subset,
        #    new_feature = params['feature'],
        #    X_f = params['sample']['X_f'],
        #    X_t = params['sample']['X_t'],
        #    y = params['sample']['y'],
        #    n_cv_ffs = int(params['n_cv_ffs']),
        #    n_jobs = n_jobs,
        #    seeds = list(map(int, params['seeds'].split(','))),
        #    train_share = float(params['ffs_train_share']),
        #    decision_function = decision_function,
        #    score_function_components = get_log_likelihood_regularized_score_balanced_components,
        #)

    return eval_result


async def handle_feature(request: web.Request):
    params = await collect_params(request)
    result = await evaluation_worker(params)
    return web.Response(text=json.dumps({'result': result}))


app = web.Application()
app.add_routes([web.post('/', handle_feature)])


if __name__ == '__main__':
    os.makedirs(TEMPORARY_FILE_FOLDER, exist_ok=True)
    web.run_app(app, host='0.0.0.0', port=5001)