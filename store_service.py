import os
import json

from aiohttp import web

STORAGE_FILE_FOLDER = './storage/'


async def store_feature(request: web.Request):
    params = await request.post()
    params_keys = list(params.keys())

    if ('key' not in params_keys) or ('feature' not in params_keys) or ('result' not in params_keys):
        raise web.HTTPError

    process_key = params['key']
    os.makedirs(STORAGE_FILE_FOLDER + f'{process_key}/', exist_ok=True)
    with open(STORAGE_FILE_FOLDER + f'{process_key}/' + params['feature'] + '.json', 'w') as f:
        f.write(params['result'])

    raise web.HTTPCreated


async def retrieve_features(request: web.Request):
    params = await request.text()
    if len(params.split('&')) > 1:
        raise web.HTTPError

    kv = params.split('=')
    if kv[0] != 'key':
        raise web.HTTPError

    key = kv[1]

    if os.path.exists(f'{STORAGE_FILE_FOLDER}' + key):
        features = dict()

        for feature in os.listdir(f'{STORAGE_FILE_FOLDER}' + key):
            feature_file = f'{STORAGE_FILE_FOLDER}' + key + '/' + feature
            if os.path.isfile(feature_file):
                with open(feature_file, 'r') as f:
                    features[''.join(feature.split('.')[:-1])] = json.loads(f.read())

        return web.Response(text=json.dumps({'result': features}))
    else:
        raise web.HTTPNoContent


app = web.Application()
app.add_routes([
    web.post('/store', store_feature),
    web.get('/retrieve', retrieve_features)
])


if __name__ == '__main__':
    os.makedirs(STORAGE_FILE_FOLDER, exist_ok=True)
    web.run_app(app, host='0.0.0.0', port=5002)