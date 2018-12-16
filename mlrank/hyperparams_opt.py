from hyperopt import fmin, tpe, hp
from sklearn.model_selection import cross_val_score, KFold
from bayes_opt import BayesianOptimization

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# hyperopt for lightgbm shows terrible results
def hyperopt_optimization_lightgbm(X, y, cv=6, max_iter_opt=15):
    space = hp.choice('clr_type', [
        {
            'type': 'lightgbm',
            'feature_fraction': hp.uniform('feature_fraction', 0.05, 0.95),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.05, 0.95),
            'bagging_freq': hp.uniform('bagging_freq', 1, 50),
            'n_estimators': hp.uniform('n_estimators', 5, 50),
        }
    ])

    best = fmin(
        fn=lambda args: cross_val_score(
            LGBMClassifier(
                boosting_type='rf',
                feature_fraction=args['feature_fraction'],
                bagging_freq=int(args['bagging_freq']),
                bagging_fraction=args['bagging_fraction'],
                n_estimators=int(args['n_estimators'])
            ),
            X, y.squeeze(), cv=KFold(n_splits=cv).split(X), scoring='accuracy'
        ).mean(),
        space=space,
        algo=tpe.suggest,
        max_evals=max_iter_opt
    )

    return best


def bayesian_optimization_lightgbm(X, y, cv=6, max_iter_opt=15):
    svr_opt = BayesianOptimization(
        lambda colsample_bytree, subsample_freq, subsample, n_estimators: cross_val_score(
            LGBMClassifier(
                boosting_type='rf',
                colsample_bytree=colsample_bytree,
                subsample_freq=int(subsample_freq),
                subsample=subsample,
                n_estimators=int(n_estimators),
                verbose=-1,
                silent=True
            ),
            X, y.squeeze(), cv=KFold(n_splits=cv).split(X), scoring='accuracy'
        ).mean(),
        {'colsample_bytree': (0.05, 0.95),
         'subsample': (0.05, 0.95),
         'subsample_freq': (1, 50),
         'n_estimators': (5, 50)},
         verbose=0
    )
    svr_opt.maximize(
        init_points=10,
        n_iter=max_iter_opt
    )
    return svr_opt.max['params']  # ['C']


def get_optimized_lightgbm(X, y):
    params_opt = bayesian_optimization_lightgbm(X, y, cv=4, max_iter_opt=8)
    params_opt['subsample_freq'] = int(params_opt['subsample_freq'])
    params_opt['n_estimators'] = int(params_opt['n_estimators'])
    return LGBMClassifier(boosting_type='rf', min_child_samples=10, **params_opt)


def get_optimized_logistic_regression(X, y):
    return LogisticRegression(random_state=42, multi_class='ovr', solver='liblinear', C=10000, tol=1e-2)


def get_optimized_svc(X, y):
    return LinearSVC(multi_class='ovr', C=10000, tol=1e-2, max_iter=250)
