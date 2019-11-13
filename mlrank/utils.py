import numpy as np
from sklearn.model_selection import train_test_split


def make_features_matrix(features: dict, subset: list):
    feature_list = [features[s] for s in subset]

    return np.hstack(feature_list)


def split_dataset(X_plain: dict, X_transformed: dict, y: np.array, seed: int, test_size) -> dict:
    train_indices, test_indices = train_test_split(range(y.size), test_size=test_size, random_state=seed)

    result = {
        'train': dict(),
        'test': dict()
    }

    result['train']['plain'] = {k: v[train_indices] for k, v in X_plain.items()}
    result['train']['transformed'] = {k: v[train_indices] for k, v in X_transformed.items()}
    result['train']['target'] = y[train_indices]

    result['test']['plain'] = {k: v[test_indices] for k, v in X_plain.items()}
    result['test']['transformed'] = {k: v[test_indices] for k, v in X_transformed.items()}
    result['test']['target'] = y[test_indices]

    return result
