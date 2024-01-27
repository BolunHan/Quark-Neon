import json

import numpy as np

__all__ = ['Scaler']


class Scaler(object):
    def __init__(self, sample_range: list[float] = None):
        self.mean = None
        self.variance = None

        self.sample_range = [0.05, 0.95] if sample_range is None else sample_range

    def fit(self, x: np.ndarray):
        # Calculate mean and variance for each feature using the center 90% of the data
        lower_percentile = self.sample_range[0]
        upper_percentile = self.sample_range[1]

        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        num_features = x.shape[1]
        mean = []
        variance = []

        for i in range(num_features):
            feature_data = x[:, i]
            is_dummy = np.all((feature_data == 0) | (feature_data == 1) | (feature_data == -1))

            if is_dummy:
                feature_mean = 0.
                feature_variance = 1
            else:
                lower_bound = np.quantile(feature_data, lower_percentile)
                upper_bound = np.quantile(feature_data, upper_percentile)
                feature_data_centered = feature_data[(feature_data >= lower_bound) & (feature_data <= upper_bound)]

                feature_mean = np.nanmean(feature_data_centered)
                feature_variance = np.nanvar(feature_data_centered, ddof=1)

                # fallback to min max scaling
                if not feature_variance:
                    feature_variance = max(feature_data) - min(feature_data)

            mean.append(feature_mean)
            variance.append(feature_variance)

        self.mean = np.array(mean)
        self.variance = np.array(variance)

    def transform(self, x, reverse: bool = False):
        single_obs = False
        transformed_columns = []

        # Transform each feature (column) of x
        if not self.is_ready:
            raise ValueError("Scaler has not been fitted. Call fit() before transform().")

        if len(x.shape) == 1:
            single_obs = True
            x = x.reshape(1, -1)

        num_features = x.shape[1]

        if not (len(self.mean) == len(self.variance) == num_features):
            raise ValueError(f'Expect x to have {len(self.mean)} features, got {num_features}, shape not match!')

        for i in range(num_features):
            feature_data = x[:, i]

            if reverse:
                transformed_column = feature_data * np.sqrt(self.variance[i]) + self.mean[i]
                transformed_columns.append(transformed_column)
            else:
                transformed_column = (feature_data - self.mean[i]) / np.sqrt(self.variance[i])
                transformed_columns.append(transformed_column)

        transformed_x = np.column_stack(transformed_columns)

        if single_obs:
            return transformed_x[0]

        return transformed_x

    def reverse_transform(self, x):
        return self.transform(x=x, reverse=True)

    def to_json(self, fmt='dict') -> dict | str:
        """
        Serialize the model to a JSON format.

        Args:
            fmt (str): Format for serialization ('dict' or 'json').

        Returns:
            dict or str: Serialized model.
        """
        json_dict = dict(
            mean=self.mean.tolist() if self.mean is not None else None,
            variance=self.variance.tolist() if self.variance is not None else None,
            sample_range=self.sample_range
        )

        if fmt == 'dict':
            return json_dict
        else:
            return json.dumps(json_dict)

    @classmethod
    def from_json(cls, json_str: str | bytes | dict):
        """
        Deserialize the model from a JSON format.

        Args:
            json_str (str, bytes, or dict): Serialized model.

        Returns:
            LinearRegression: Deserialized model.
        """
        if isinstance(json_str, (str, bytes)):
            json_dict = json.loads(json_str)
        elif isinstance(json_str, dict):
            json_dict = json_str
        else:
            raise TypeError(f'{cls.__name__} can not load from json {json_str}')

        self = cls(sample_range=json_dict['sample_range'])

        self.mean = np.array(json_dict["mean"]) if json_dict["mean"] is not None else None
        self.variance = np.array(json_dict["variance"]) if json_dict["variance"] is not None else None

        return self

    @property
    def is_ready(self) -> bool:
        if self.mean is None or self.variance is None:
            return False

        return True
