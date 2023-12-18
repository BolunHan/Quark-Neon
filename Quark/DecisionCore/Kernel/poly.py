import numpy as np
import pandas as pd


def poly_features(data: pd.DataFrame | dict[str, list | np.ndarray], degree: int = 3) -> dict[str, np.ndarray]:
    names = data.keys()

    for _ in names:
        if ' * ' in _:
            raise KeyError(f'The data can not have " * " in this keys or columns. Invalid name: {_}')

    extended_names = [[_] for _ in names]
    extended_features = {}

    if degree <= 1:
        return data

    for i in range(1, degree):  # 0, 1, 2
        extended_orders = []
        for new_term in names[:]:  # x1
            t = extended_names[:]  # [['x1'], ['x2'], ['x3']]
            for original_term in t:
                extended_orders.append(original_term + [new_term])  # [['x1', 'x1'], ['x2', 'x1'], ['x3', 'x1']]
        extended_names = extended_orders
    extended_names = sorted(set([' * '.join(sorted(_)) for _ in extended_names]))

    for names in extended_names:
        features = names.split(' * ')
        value = 1

        for feature in features:
            _ = np.array(data[feature])
            value *= _

        extended_features[names] = value

    return extended_features
