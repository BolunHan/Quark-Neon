"""
This module defines a function, poly_features, for generating polynomial features from input data.

Functions:
- poly_features: Generates polynomial features from input data.

Usage:
1. Call poly_features with the input data and optional parameters.
2. Retrieve the generated polynomial features.

Note: This module assumes the availability of NumPy and pandas.

Author: Bolun
Date: 2023-12-27
"""

from collections import Counter

import numpy as np
import pandas as pd


def poly_features(
        data: pd.DataFrame | dict[str, list | np.ndarray | int | float],
        degree: int = 3,
        simplified_name: bool = True
) -> dict[str, np.ndarray | int | float]:
    """
    Generate polynomial features from input data.

    Args:
    - data (pd.DataFrame | dict[str, list | np.ndarray | int | float]): Input data.
    - degree (int, optional): Degree of the polynomial. Defaults to 3.
    - simplified_name (bool, optional): Whether to simplify feature names. Defaults to True.

    Returns:
    dict[str, np.ndarray | int | float]: Dictionary of generated polynomial features.
    """
    expression = list(data.keys())

    for _ in expression:
        if ' * ' in _:
            raise KeyError(f'The data cannot have " * " in these keys or columns. Invalid name: {_}')

    extended_names = [[_] for _ in expression]
    extended_features = {}

    if degree <= 1:
        return data

    for i in range(1, degree):  # 0, 1, 2
        extended_orders = []
        for new_term in expression[:]:  # x1
            t = extended_names[:]  # [['x1'], ['x2'], ['x3']]
            for original_term in t:
                extended_orders.append(original_term + [new_term])  # [['x1', 'x1'], ['x2', 'x1'], ['x3', 'x1']]
        extended_names = extended_orders
    extended_names = sorted(set([' * '.join(sorted(_)) for _ in extended_names]))

    for expression in extended_names:
        features = expression.split(' * ')
        sorted_terms = sorted(features)
        term_counts = Counter(sorted_terms)

        expression_simplified = ' * '.join([f'{term}^{count}' if count > 1 else term for term, count in term_counts.items()])

        # check if the expression has single feature and the feature is a dummy.
        if len(term_counts) == 1:
            feature = list(term_counts)[0]
            feature_data = np.array(data[feature])
            is_dummy = np.all((feature_data == 0) | (feature_data == 1))

            if is_dummy:
                continue

        if simplified_name:
            expression = expression_simplified

        value = 1
        all_dummy = True

        for feature in features:
            feature_data = np.array(data[feature])
            is_dummy = np.all((feature_data == 0) | (feature_data == 1))

            if not is_dummy:
                all_dummy = False

            value = np.multiply(value, feature_data)

        # the expression has only the dummies and they excluded each other
        if all_dummy and len(np.unique(value)) == 1:
            continue

        extended_features[expression] = value

    return extended_features
