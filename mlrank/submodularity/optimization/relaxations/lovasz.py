import numpy as np

from functools import partial


def lovasz(set_function):
    def make_relaxed(R, V):
        set_order = np.argsort(R)
        set_ordered = V[set_order]

        set_function_values = [set_function(set_ordered[0:i]) for i in range(0, V.shape[0])]

        support = np.diff(set_function_values)[::-1]
        support = np.append(support, [0])[::-1]

        return support

    return make_relaxed


def bind_subset(relaxed_function, subset):
    return partial(relaxed_function, V = subset)
