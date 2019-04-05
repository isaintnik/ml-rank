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


def bayesian_optimization_lightgbm(X, y, cv=6, max_iter_opt=15, decision_function='gbdt', expra_params:dict=dict()):
    svr_opt = BayesianOptimization(
        lambda learning_rate, max_depth: cross_val_score(
            LGBMClassifier(
                boosting_type=decision_function,
                learning_rate=learning_rate,
                max_depth=int(max_depth),
                subsample=0.7,
                n_estimators=200,
                verbose=-1,
                subsample_freq=5, # ????
                num_leaves=2**int(max_depth),
                silent=True,
                **expra_params
            ),
            X, y.squeeze(), cv=KFold(n_splits=cv).split(X), scoring='accuracy'
        ).mean(),
        {
         # глубина?
         'max_depth': (3, 7),
         'learning_rate': (1e-3, 0.05)
        },
         verbose=0
    )
    svr_opt.maximize(
        init_points=10,
        n_iter=max_iter_opt
    )
    return svr_opt.max['params']  # ['C']


def get_optimized_lightgbm_gbdt(X, y):
    params_opt = bayesian_optimization_lightgbm(X, y, cv=4, max_iter_opt=8)
    params_opt['max_depth'] = int(params_opt['max_depth'])
    params_opt['subsample'] = 0.7
    params_opt['n_estimators'] = 200
    params_opt['subsample_freq'] = 5
    params_opt['num_leaves'] = int(params_opt['max_depth']) ** 2
    params_opt['boosting_type'] = 'gbdt'
    return LGBMClassifier(**params_opt)


def get_optimized_lightgbm_rf(X, y):
    params_opt = bayesian_optimization_lightgbm(X, y, cv=4, max_iter_opt=8, decision_function='rf', expra_params={'colsample_bytree': .2})
    params_opt['max_depth'] = int(params_opt['max_depth'])
    params_opt['subsample'] = 0.7
    params_opt['n_estimators'] = 200
    params_opt['subsample_freq'] = 5
    params_opt['num_leaves'] = int(params_opt['max_depth']) ** 2
    params_opt['boosting_type'] = 'rf'
    params_opt['colsample_bytree'] = .2
    return LGBMClassifier(**params_opt)


def get_optimized_logistic_regression(X, y):
    return LogisticRegression(random_state=42, multi_class='ovr', solver='liblinear', C=10000, tol=1e-2)


def get_optimized_svc(X, y):
    return LinearSVC(multi_class='ovr', C=10000, tol=1e-2, max_iter=250)
