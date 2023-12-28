from typing import Hashable
from functools import cached_property
import numpy as np

from . import Regression


class Cache(object):
    def __init__(self):
        self.cache = {}

    def __call__(self, func):

        def wrapper(*args, **kwargs):
            entry_list = []

            for _ in args:
                if isinstance(_, Hashable):
                    entry_list.append(_)
                else:
                    entry_list.append(id(_))

            for _ in kwargs.values():
                if isinstance(_, Hashable):
                    entry_list.append(_)
                else:
                    entry_list.append(id(_))

            cache_key = tuple(entry_list)

            if cache_key in self.cache:
                value = self.cache[cache_key]
            else:
                value = self.cache[cache_key] = func(*args, **kwargs)

            return value

        return wrapper


METRIC_CACHE = Cache()


class Metrics(object):
    def __init__(self, model: Regression, x: np.ndarray, y: np.ndarray):
        self.model = model
        self.x = x
        self.y = y

    def __del__(self):
        self.model = None
        self.x = None
        self.y = None
        METRIC_CACHE.cache.clear()

    @cached_property
    def metrics(self):
        alpha_range = np.linspace(0.01, 0.5, 50)
        y_pred, prediction_interval, *_ = self._predict(model=self.model, x=self.x, alpha=0.95)
        mse = self.compute_mse(y_actual=self.y, y_pred=y_pred)
        mae = self.compute_mae(y_actual=self.y, y_pred=y_pred)
        accuracy = self.compute_accuracy(y_actual=self.y, y_pred=y_pred)

        mse_significant = self.compute_mse_significant(model=self.model, x=self.x, y_actual=self.y, alpha=0.95)
        mae_significant = self.compute_mae_significant(model=self.model, x=self.x, y_actual=self.y, alpha=0.95)
        accuracy_significant = self.compute_accuracy_significant(model=self.model, x=self.x, y_actual=self.y, alpha=0.95)

        _, roc_values = self.compute_roc(model=self.model, x=self.x, y_actual=self.y, alpha_range=alpha_range)

        return dict(
            mse=mse,
            mae=mae,
            accuracy=accuracy,
            mse_significant=mse_significant,
            mae_significant=mae_significant,
            accuracy_significant=accuracy_significant,
            alpha_range=alpha_range,
            roc_values=roc_values
        )

    @classmethod
    @METRIC_CACHE
    def _predict(cls, model: Regression, x: np.ndarray, alpha: float):
        return model.predict(x=x, alpha=alpha)

    @classmethod
    @METRIC_CACHE
    def _select_significant_prediction(cls, model: Regression, x: np.ndarray, y_actual: np.ndarray, alpha: float):
        y_pred, prediction_interval, *_ = model.predict(x=x, alpha=alpha)
        lower_bound = y_pred + prediction_interval[:, 0]
        upper_bound = y_pred + prediction_interval[:, 1]

        # Select data where lower bound >= 0 or upper bound <= 0
        selected_indices = (lower_bound >= 0) | (upper_bound <= 0)

        if not np.any(selected_indices):
            return None, None

        y_actual_selected = y_actual[selected_indices]
        y_pred_selected = y_pred[selected_indices]

        return y_actual_selected, y_pred_selected

    @classmethod
    def compute_mse(cls, y_actual: np.ndarray, y_pred: np.ndarray):
        residuals = y_actual - y_pred
        mse = np.mean(residuals ** 2)
        return mse

    @classmethod
    def compute_mae(cls, y_actual: np.ndarray, y_pred: np.ndarray):
        residuals = y_actual - y_pred
        mae = np.mean(np.abs(residuals))
        return mae

    @classmethod
    def compute_accuracy(cls, y_actual: np.ndarray, y_pred: np.ndarray):
        """
        Compute accuracy as the percentage of correct sign predictions.

        Args:
            y_actual (numpy.ndarray): Actual output values.
            y_pred (numpy.ndarray): Predicted output values.

        Returns:
            float: Accuracy.
        """
        correct_signs = np.sign(y_actual) == np.sign(y_pred)
        accuracy = np.mean(correct_signs)
        return accuracy

    @classmethod
    def compute_mse_significant(cls, model: Regression, x: np.ndarray, y_actual: np.ndarray, alpha: float):
        y_actual, y_pred = cls._select_significant_prediction(model=model, x=x, y_actual=y_actual, alpha=alpha)
        mse_significant = np.mean((y_actual - y_pred) ** 2) if y_actual is not None else np.nan
        return mse_significant

    @classmethod
    def compute_mae_significant(cls, model: Regression, x: np.ndarray, y_actual: np.ndarray, alpha: float):
        y_actual, y_pred = cls._select_significant_prediction(model=model, x=x, y_actual=y_actual, alpha=alpha)
        mae_significant = np.mean(np.abs(y_actual - y_pred)) if y_actual is not None else np.nan
        return mae_significant

    @classmethod
    def compute_accuracy_significant(cls, model: Regression, x: np.ndarray, y_actual: np.ndarray, alpha: float):
        y_actual, y_pred = cls._select_significant_prediction(model=model, x=x, y_actual=y_actual, alpha=alpha)
        accuracy_significant = np.mean(np.sign(y_actual) == np.sign(y_pred)) if y_actual is not None else np.nan
        return accuracy_significant

    @classmethod
    def compute_roc(cls, model: Regression, x: np.ndarray, y_actual: np.ndarray, alpha_range=np.linspace(0.01, 0.5, 50)):

        roc_values = []
        for alpha in alpha_range:
            accuracy = cls.compute_accuracy_significant(model=model, x=x, y_actual=y_actual, alpha=alpha)
            roc_values.append(accuracy)

        return alpha_range, np.array(roc_values)


class CrossValidation(object):
    def __init__(self, model, folds=5, strict_no_future: bool = True, shuffle: bool = True, **kwargs):
        """
        Initialize the CrossValidation object.

        Args:
            model: Regression model object (e.g., BootstrapLinearRegression).
            folds (int): Number of folds for cross-validation.
            strict_no_future (bool): training data must be prier to ALL the validation data
            shuffle (bool): shuffle the training and validation index
        """
        self.model = model
        self.folds = folds
        self.strict_no_future = strict_no_future
        self.shuffle = shuffle

        self.fit_kwargs = kwargs

        self.x_val = None
        self.y_val = None
        self.y_pred = None
        self.prediction_interval = None

    @classmethod
    def _select_data(cls, x: np.ndarray, y: np.ndarray, indices: np.ndarray, fold: int, n_folds: int, shuffle: bool = False):
        n = len(x)
        start_idx = n // n_folds * fold
        end_idx = n // n_folds * (fold + 1) if fold < n_folds - 1 else n

        val_indices = indices[start_idx:end_idx].copy()
        train_indices = np.setdiff1d(indices, val_indices).copy()

        if shuffle:
            np.random.shuffle(val_indices)
            np.random.shuffle(train_indices)

        x_train, y_train, x_val, y_val = x[train_indices], y[train_indices], x[val_indices], y[val_indices]

        return x_train, y_train, x_val, y_val, val_indices

    @classmethod
    def _select_data_sequential(cls, x: np.ndarray, y: np.ndarray, indices: np.ndarray, fold: int, n_folds: int, shuffle: bool = False):
        n = len(x)
        if fold == 0:
            start_idx = n // (n_folds * 2)
            end_idx = n // n_folds
        else:
            start_idx = n // n_folds * fold
            end_idx = n // n_folds * (fold + 1) if fold < n_folds - 1 else n

        train_indices = indices[0: start_idx].copy()
        val_indices = indices[start_idx:end_idx].copy()

        if shuffle:
            np.random.shuffle(val_indices)
            np.random.shuffle(train_indices)

        x_train, y_train, x_val, y_val = x[train_indices], y[train_indices], x[val_indices], y[val_indices]
        return x_train, y_train, x_val, y_val, val_indices

    def validate(self, x: np.ndarray, y: np.ndarray):
        """
        Perform cross-validation and store the results in the metrics attribute.

        Args:
            x (numpy.ndarray): Input features.
            y (numpy.ndarray): Output values.

        Returns:
            None
        """
        n = len(x)
        indices = np.arange(n)

        if not self.strict_no_future:
            np.random.shuffle(indices)

        fold_metrics = {'x': [], 'y_val': [], 'y_pred': [], 'index': [], 'prediction_interval': []}

        for fold in range(self.folds):
            if self.strict_no_future:
                x_train, y_train, x_val, y_val, val_indices = self._select_data_sequential(x=x, y=y, indices=indices, fold=fold, n_folds=self.folds, shuffle=self.shuffle)
            else:
                x_train, y_train, x_val, y_val, val_indices = self._select_data(x=x, y=y, indices=indices, fold=fold, n_folds=self.folds, shuffle=self.shuffle)

            self.model.fit(x=x_train, y=y_train, **self.fit_kwargs)

            # Predict on the validation data
            y_pred, prediction_interval, *_ = self.model.predict(x_val)

            fold_metrics['x_val'].append(x_val)
            fold_metrics['y_val'].append(y_val)
            fold_metrics['y_pred'].append(y_pred)
            fold_metrics['index'].append(val_indices)  # Store sorted indices
            fold_metrics['prediction_interval'].append(prediction_interval)

        sorted_indices = np.concatenate(fold_metrics['index'])
        for key in ['x_val', 'y_val', 'y_pred', 'prediction_interval']:
            values = np.concatenate(fold_metrics[key])
            fold_metrics[key] = values[np.argsort(sorted_indices)]

        self.x_val = fold_metrics['x_val']
        self.y_val = fold_metrics['y_val']
        self.y_pred = fold_metrics['y_pred']
        self.prediction_interval = fold_metrics['prediction_interval']

    def validate_out_sample(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray):
        self.model.fit(x=x_train, y=y_train, **self.fit_kwargs)

        # Predict on the validation data
        y_pred, prediction_interval, *_ = self.model.predict(x_val)

        self.x_val = np.array(x_val)
        self.y_val = np.array(y_val)
        self.y_pred = np.array(y_pred)
        self.prediction_interval = np.array(prediction_interval)

    def plot(self, x_axis: list | np.ndarray = None, **kwargs):
        """
        Plot the validation y_pred and y_val using Plotly.

        Args:
            x_axis (numpy.ndarray, optional): Values for the x-axis. Default is None.

        Returns:
            plotly.graph_objects.Figure: Plotly figure object.
        """
        import plotly.graph_objects as go

        fig = go.Figure()
        y_pred = self.y_pred
        y_val = self.y_val
        prediction_interval = self.prediction_interval

        if x_axis is None:
            x_axis = np.arange(len(y_val))

        # Scatter plot for actual values
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=y_val,
                mode='markers',
                name=kwargs.get('data_name', "y_val"),
                yaxis='y1'
            )
        )

        # Line plot for predicted values
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=y_pred,
                mode='lines',
                name=kwargs.get('model_name', "y_pred"),
                line=dict(color='red'),
                yaxis='y1'
            )
        )

        # Plot prediction interval (shadow area)
        if prediction_interval is not None and np.any(prediction_interval):
            fig.add_trace(
                go.Scatter(
                    name=f'Upper Bound',
                    x=x_axis,
                    y=y_pred + prediction_interval[:, 1],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    showlegend=False,
                    yaxis='y1'
                )
            )

            fig.add_trace(
                go.Scatter(
                    name=f'Lower Bound',
                    x=x_axis,
                    y=y_pred + prediction_interval[:, 0],
                    line=dict(color='red', dash='dash'),
                    mode='lines',
                    fillcolor='rgba(255,0,0,0.3)',
                    fill='tonexty',
                    showlegend=False,
                    yaxis='y1'
                )
            )

        # Layout settings
        fig.update_layout(
            title=kwargs.get('title', "Cross-Validation: Actual vs Predicted"),
            xaxis_title=kwargs.get('x_name', "Index"),
            yaxis_title=kwargs.get('y_name', "Values"),
            hovermode="x unified",
            template='simple_white',
            yaxis=dict(
                showspikes=True
            ),
        )

        return fig

    @cached_property
    def metrics(self) -> dict[str, float] | None:
        if self.x_val is None:
            return None

        metrics = Metrics(
            model=self.model,
            x=self.x_val,
            y=self.y_val
        )

        return metrics.metrics
