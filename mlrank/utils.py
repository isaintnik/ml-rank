import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import safe_indexing, safe_mask


def make_features_matrix(features: dict, subset: list):
    feature_list = [features[s] for s in subset]

    return np.vstack(feature_list).T


def split_dataset(X_plain: dict, X_transformed: dict, y: np.array, seed: int, test_size) -> dict:
    train_indices, test_indices = train_test_split(range(y.size), test_size=test_size, random_state=seed)

    result = {
        'train': dict(),
        'test': dict()
    }

    #result['train']['plain'] = {k: safe_indexing(v, train_indices) for k, v in X_plain.items()}
    result['train']['plain'] = {k: np.take(v, train_indices, axis=-1) for k, v in X_plain.items()}
    #result['train']['transformed'] = {k: safe_indexing(v, train_indices) for k, v in X_transformed.items()}
    result['train']['transformed'] = {k: np.take(v, train_indices, axis=-1) for k, v in X_transformed.items()}
    #result['train']['target'] = safe_indexing(y, train_indices)
    result['train']['target'] = np.take(y, train_indices, axis=-1)

    #result['test']['plain'] = {k: safe_indexing(v, test_indices) for k, v in X_plain.items()}
    result['test']['plain'] = {k: np.take(v, test_indices, axis=-1) for k, v in X_plain.items()}
    #result['test']['transformed'] = {k: safe_indexing(v, test_indices) for k, v in X_transformed.items()}
    result['test']['transformed'] = {k: np.take(v, test_indices, axis=-1) for k, v in X_transformed.items()}
    #result['test']['target'] = safe_indexing(y, test_indices)
    result['test']['target'] = np.take(y, test_indices, axis=-1)

    return result


def get_model_classification_order(model):
    if True:
        return model.classes_

    raise Exception('model not supported')


def fix_target(classes_, target_: np.array, pred_: np.array):
    if not np.array_equal(classes_, np.arange(len(classes_))):
        for i_, c_ in enumerate(classes_):
            target_[target_ == c_] = -i_
        target_ *= -1

    return safe_indexing(target_, np.where(target_ >= 0)[0]), safe_indexing(pred_, np.where(target_ >= 0)[0])
    #return target_, pred_
