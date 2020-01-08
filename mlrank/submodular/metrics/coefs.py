from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def get_model_n_coefs(model):
    if type(model) == LogisticRegression:
        return model.coef_.shape[0] * model.coef_.shape[1]
    elif type(model) == LGBMClassifier:
        return model.booster_.num_trees()
    elif type(model) == MLPClassifier:
        return sum(i.size for i in model.coefs_) + sum(i.size for i in model.intercepts_)
    else:
        raise Exception(f'Unsupported type of model {str(type(model))}. Don\'t know how to get number of parameters.')
