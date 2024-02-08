import pathlib
from functools import cached_property
from typing import Hashable

import numpy as np
import pandas as pd

from . import Regression, LOGGER
from .Boosting import Boosting
from .Linear import LinearBootstrap, LinearRegression
from .kelly import kelly_bootstrap
from ..Base import GlobalStatics

RANGE_BREAK = GlobalStatics.RANGE_BREAK


class Cache(object):
    """
    Cache decorator for memoization.
    """

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
    """
    Metrics class for evaluating regression model performance.
    """

    def __init__(self, model: Regression, x: np.ndarray, y: np.ndarray, informed_baseline: bool = False, cost: float = 0.):
        """
        Initialize Metrics class.

        Args:
            model (Regression): Regression model object.
            x (numpy.ndarray): Input features.
            y (numpy.ndarray): Output values.
            informed_baseline (bool): True if the distribution of the y_val is known by the model.
        """
        self.model = model
        self.x = x
        self.y = y

        self.alpha = 0.05
        self.alpha_range = np.linspace(0., 1, 100)
        self.informed_baseline = informed_baseline  # True if the distribution of the y_val is known by the model.
        self.cost = cost  # trading cost of the metrics

    def __del__(self):
        """
        Destructor to clean up the object.
        """
        self.model = None
        self.x = None
        self.y = None
        METRIC_CACHE.cache.clear()

    @cached_property
    def metrics(self) -> dict[str, float]:
        """
        Calculate and return various regression metrics.

        Returns:
            dict: Dictionary containing calculated metrics.
        """
        y_pred, prediction_interval, *_ = self._predict(model=self.model, x=self.x, alpha=self.alpha)

        if self.informed_baseline:
            accuracy_baseline = 0.5 + np.abs(np.mean(np.sign(self.y))) / 2
        else:
            accuracy_baseline = 0.5

        mse = self.compute_mse(y_actual=self.y, y_pred=y_pred)
        mae = self.compute_mae(y_actual=self.y, y_pred=y_pred)
        accuracy = self.compute_accuracy(y_actual=self.y, y_pred=y_pred)

        mse_significant, _ = self.compute_mse_significant(model=self.model, x=self.x, y_actual=self.y, alpha=self.alpha)
        mae_significant, _ = self.compute_mae_significant(model=self.model, x=self.x, y_actual=self.y, alpha=self.alpha)
        accuracy_significant, selection_ratio = self.compute_accuracy_significant(model=self.model, x=self.x, y_actual=self.y, alpha=self.alpha)

        auc_roc = self.compute_auc_roc(model=self.model, x=self.x, y_actual=self.y, alpha_range=self.alpha_range) - accuracy_baseline
        kelly_return = self.compute_kelly_return(model=self.model, x=self.x, y_actual=self.y, cost=self.cost)

        return dict(
            obs_num=len(self.x),
            mse=mse,
            mae=mae,
            accuracy_baseline=accuracy_baseline,
            accuracy_boost=accuracy - accuracy_baseline,
            mse_significant=mse_significant,
            mae_significant=mae_significant,
            accuracy_boost_significant=accuracy_significant - accuracy_baseline,
            significant_ratio=selection_ratio,
            auc_roc=auc_roc,
            kelly_return=kelly_return
        )

    @classmethod
    @METRIC_CACHE
    def _predict(cls, model: Regression, x: np.ndarray, alpha: float):
        """
        Predict using the regression model.

        Args:
            model (Regression): Regression model object.
            x (numpy.ndarray): Input features.
            alpha (float): Significance level for the prediction interval.

        Returns:
            Tuple: Predicted values, prediction interval, and residuals.
        """
        return model.predict(x=x, alpha=alpha)

    @classmethod
    # @METRIC_CACHE
    def _select_significant_prediction(cls, model: Regression, x: np.ndarray, y_actual: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Select data with significant predictions based on the prediction interval.

        Args:
            model (Regression): Regression model object.
            x (numpy.ndarray): Input features.
            y_actual (numpy.ndarray): Actual output values.
            alpha (float): Significance level for the prediction interval.

        Returns:
            Tuple: Selected actual and predicted values.
        """
        y_pred, prediction_interval, *_ = cls._predict(model=model, x=x, alpha=alpha)
        lower_bound = y_pred + prediction_interval[:, 0]
        upper_bound = y_pred + prediction_interval[:, 1]

        # Select data where lower bound >= 0 or upper bound <= 0
        selected_indices = (lower_bound >= 0) | (upper_bound <= 0)

        if not np.any(selected_indices):
            return np.array([]), np.array([])

        y_actual_selected = y_actual[selected_indices]
        y_pred_selected = y_pred[selected_indices]

        return y_actual_selected, y_pred_selected

    @classmethod
    def _compute_kelly(cls, model: Regression | LinearBootstrap | Boosting, x: np.ndarray, max_pos: float = 2., cost: float = 0.):
        if model.ensemble == 'bagging':
            return cls._compute_kelly_bagging(model=model, x=x, max_pos=max_pos, cost=cost)
        elif model.ensemble == 'boosting':
            return cls._compute_kelly_boosting(model=model, x=x, max_pos=max_pos, cost=cost)
        else:
            LOGGER.error(f'Kelly function cannot handle ensemble type {model.ensemble} of model {model.__class__}.')
            return 0.

    @classmethod
    def _compute_kelly_bagging(cls, model: Regression, x: np.ndarray, max_pos: float = 2., cost: float = 0.):
        y_pred, _, bootstrap_deviation, *_ = cls._predict(model=model, x=x, alpha=0)

        kelly_value = []
        for outcome, deviations in zip(y_pred, bootstrap_deviation):
            kelly_proportion = kelly_bootstrap(outcomes=np.array(deviations) + outcome, cost=cost, max_leverage=max_pos)
            kelly_value.append(kelly_proportion)

        return np.array(kelly_value)

    @classmethod
    def _compute_kelly_boosting(cls, model: Boosting, x: np.ndarray, max_pos: float = 2., cost: float = 0., alpha_range: list[float] = None):

        outcome_array = model.resample(x=x, alpha_range=alpha_range)

        kelly_value = []
        for outcome in outcome_array:
            kelly_proportion = kelly_bootstrap(outcomes=outcome, cost=cost, max_leverage=max_pos)
            kelly_value.append(kelly_proportion)

        return np.array(kelly_value)

    @classmethod
    def compute_mse(cls, y_actual: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Mean Squared Error (MSE).

        Args:
            y_actual (numpy.ndarray): Actual output values.
            y_pred (numpy.ndarray): Predicted output values.

        Returns:
            float: MSE.
        """
        residuals = y_actual - y_pred
        mse = np.mean(residuals ** 2)
        return mse

    @classmethod
    def compute_mae(cls, y_actual: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Mean Absolute Error (MAE).

        Args:
            y_actual (numpy.ndarray): Actual output values.
            y_pred (numpy.ndarray): Predicted output values.

        Returns:
            float: MAE.
        """
        residuals = y_actual - y_pred
        mae = np.mean(np.abs(residuals))
        return mae

    @classmethod
    def compute_accuracy(cls, y_actual: np.ndarray, y_pred: np.ndarray) -> float:
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
    def compute_mse_significant(cls, model: Regression, x: np.ndarray, y_actual: np.ndarray, alpha: float) -> tuple[float, float]:
        """
        Compute MSE for significant predictions based on the prediction interval.

        Args:
            model (Regression): Regression model object.
            x (numpy.ndarray): Input features.
            y_actual (numpy.ndarray): Actual output values.
            alpha (float): Significance level for the prediction interval.

        Returns:
            float: MSE for significant predictions or np.nan if no data is left after selection.
        """
        y_actual_selected, y_pred_selected = cls._select_significant_prediction(model=model, x=x, y_actual=y_actual, alpha=alpha)
        mse_significant = np.mean((y_actual_selected - y_pred_selected) ** 2) if y_actual.size > 0 else np.nan
        selection_ratio = len(y_actual_selected) / len(y_actual) if y_actual.size > 0 else np.nan
        return mse_significant, selection_ratio

    @classmethod
    def compute_mae_significant(cls, model: Regression, x: np.ndarray, y_actual: np.ndarray, alpha: float) -> tuple[float, float]:
        """
        Compute MAE for significant predictions based on the prediction interval.

        Args:
            model (Regression): Regression model object.
            x (numpy.ndarray): Input features.
            y_actual (numpy.ndarray): Actual output values.
            alpha (float): Significance level for the prediction interval.

        Returns:
            float: MAE for significant predictions or np.nan if no data is left after selection.
        """
        y_actual_selected, y_pred_selected = cls._select_significant_prediction(model=model, x=x, y_actual=y_actual, alpha=alpha)
        mae_significant = np.mean(np.abs(y_actual_selected - y_pred_selected)) if y_actual.size > 0 else np.nan
        selection_ratio = len(y_actual_selected) / len(y_actual) if y_actual.size > 0 else np.nan
        return mae_significant, selection_ratio

    @classmethod
    def compute_accuracy_significant(cls, model: Regression, x: np.ndarray, y_actual: np.ndarray, alpha: float) -> tuple[float, float]:
        """
        Compute accuracy for significant predictions based on the prediction interval.

        Args:
            model (Regression): Regression model object.
            x (numpy.ndarray): Input features.
            y_actual (numpy.ndarray): Actual output values.
            alpha (float): Significance level for the prediction interval.

        Returns:
            float: Accuracy for significant predictions or np.nan if no data is left after selection.
        """
        y_actual_selected, y_pred_selected = cls._select_significant_prediction(model=model, x=x, y_actual=y_actual, alpha=alpha)
        accuracy_significant = np.mean(np.sign(y_actual_selected) == np.sign(y_pred_selected)) if y_actual.size > 0 else np.nan
        selection_ratio = len(y_actual_selected) / len(y_actual) if y_actual.size > 0 else np.nan
        return accuracy_significant, selection_ratio

    @classmethod
    def compute_roc(cls, model: Regression, x: np.ndarray, y_actual: np.ndarray, alpha_range: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Receiver Operating Characteristic (ROC) values.

        Args:
            model (Regression): Regression model object.
            x (numpy.ndarray): Input features.
            y_actual (numpy.ndarray): Actual output values.
            alpha_range (numpy.ndarray): Array of significance levels.

        Returns:
            Tuple: Array of alpha values and corresponding ROC values.
        """
        roc_values = []
        for alpha in alpha_range:
            accuracy, _ = cls.compute_accuracy_significant(model=model, x=x, y_actual=y_actual, alpha=alpha)
            roc_values.append(accuracy)

        return alpha_range, np.array(roc_values)

    @classmethod
    def compute_auc_roc(cls, model: Regression, x: np.ndarray, y_actual: np.ndarray, alpha_range: np.ndarray) -> float:
        """
        Compute the Area Under the Receiver Operating Characteristic (ROC) Curve (AUC-ROC).

        Args:
            model (Regression): Regression model object.
            x (numpy.ndarray): Input features.
            y_actual (numpy.ndarray): Actual output values.
            alpha_range (numpy.ndarray): Array of significance levels.

        Returns:
            float: AUC-ROC value.
        """
        roc_values = []
        for alpha in alpha_range:
            accuracy, _ = cls.compute_accuracy_significant(model=model, x=x, y_actual=y_actual, alpha=alpha)
            roc_values.append(accuracy)

        roc_values = np.array(roc_values)
        valid_indices = ~np.isnan(roc_values)

        if np.any(valid_indices):
            auc_roc = -np.trapz(roc_values[valid_indices], (1 - alpha_range)[valid_indices])
            return auc_roc
        else:
            return np.nan

    @classmethod
    def compute_kelly_return(cls, model: Regression, x: np.ndarray, y_actual: np.ndarray, max_pos: float = 2., cost: float = 0.):
        kelly_return = []

        kelly_decision = cls._compute_kelly(model=model, x=x, max_pos=max_pos, cost=cost)

        current_pos = 0.

        for weight, outcome in zip(kelly_decision, y_actual):
            if weight:
                kelly_return.append(weight * outcome)

            # if current_pos == weight:
            #     pass
            # else:
            #     trade_action = weight - current_pos
            #     current_pos = weight
            #     trade_cost = abs(trade_action) * cost
            #     trade_return = outcome * current_pos
            #     kelly_return += trade_return
            #     kelly_return -= trade_cost

        return np.nanmean(kelly_return)

    @classmethod
    def plot_roc(cls, model: Regression, x: np.ndarray, y_actual: np.ndarray, alpha_range: np.ndarray, accuracy_baseline: float = 0., **kwargs):
        """
        Plot the Receiver Operating Characteristic (ROC) Curve.

        Args:
            model (Regression): Regression model object.
            x (numpy.ndarray): Input features.
            y_actual (numpy.ndarray): Actual output values.
            alpha_range (numpy.ndarray): Array of significance levels.
            accuracy_baseline: Baseline value of accuracy
            **kwargs: Additional keyword arguments for customizing the plot.

        Returns:
            plotly.graph_objects.Figure: Plotly figure object.
        """
        import plotly.graph_objects as go

        roc_values = []
        selection_ratios = []

        for alpha in alpha_range:
            accuracy, selection_ratio = cls.compute_accuracy_significant(model=model, x=x, y_actual=y_actual, alpha=alpha)
            roc_values.append(accuracy - accuracy_baseline)
            selection_ratios.append(selection_ratio)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=1 - alpha_range,
                y=roc_values,
                mode='lines',
                name=kwargs.get('curve_name', "ROC Curve"),
                line=dict(color='blue'),
                yaxis='y1'
            )
        )

        fig.add_trace(
            go.Bar(
                x=1 - alpha_range,
                y=selection_ratios,
                opacity=0.3,
                name="Selection Ratio",
                marker=dict(color='green'),
                yaxis='y2'
            )
        )

        fig.update_layout(
            title=kwargs.get('title', "Receiver Operating Characteristic (ROC) Curve"),
            xaxis_title=kwargs.get('x_name', "Classification threshold (1 - alpha)"),
            yaxis_title=kwargs.get('y_name', "Accuracy Boost" if accuracy_baseline else "Accuracy"),
            hovermode="x unified",
            template='simple_white',
            showlegend=False,
            yaxis=dict(
                maxallowed=0.5 if accuracy_baseline else 1,
                minallowed=-0.5 if accuracy_baseline else 0,
                showspikes=True,
                tickformat='.2%'
            ),
            yaxis2=dict(
                title="Selection Ratio",
                range=[0, 1],
                overlaying='y',
                side='right',
                showgrid=False,
                tickformat='.2%'
            )
        )

        return fig

    def to_html(self, file_path: str | pathlib.Path):
        """
        Export metrics and ROC curve plot to an HTML file.

        Args:
            file_path (str): File path for the HTML file.

        Returns:
            None
        """
        metrics_data = self.metrics.copy()

        # Create metrics table figure
        # noinspection PyTypeChecker
        metrics_data['obs_num'] = f"{metrics_data['obs_num']:,d}"
        metrics_table = pd.DataFrame({'Metrics': pd.Series(metrics_data)})

        # Create ROC curve figure
        roc_curve = self.plot_roc(
            model=self.model,
            x=self.x,
            y_actual=self.y,
            accuracy_baseline=metrics_data['accuracy_baseline'],
            alpha_range=self.alpha_range
        )

        # Convert the figures to HTML codes
        metrics_table_html = metrics_table.to_html(float_format=lambda x: f'{x:.4%}')
        roc_curve_html = roc_curve.to_html(roc_curve, full_html=False)

        # Create a 1x2 table HTML code
        html_code = f"""
        <html>
        <head></head>
        <body>
            <table style="width:100%">
                <tr>
                    <td style="width:30%">{metrics_table_html}</td>
                    <td style="width:70%">{roc_curve_html}</td>
                </tr>
            </table>
        </body>
        </html>
        """

        # Write the HTML code to the file
        with open(file_path, 'w') as file:
            file.write(html_code)


class CrossValidation(object):
    def __init__(self, model: Regression, folds=5, strict_no_future: bool = True, shuffle: bool = True, trade_cost: float = 0.001, **kwargs):
        """
        Initialize the CrossValidation object.

        Args:
            model: Regression model object (e.g., LinearRegression).
            folds (int): Number of folds for cross-validation.
            strict_no_future (bool): training data must be prier to ALL the validation data
            shuffle (bool): shuffle the training and validation index
        """
        self.model = model
        self.folds = folds
        self.strict_no_future = strict_no_future
        self.shuffle = shuffle
        self.trade_cost = trade_cost

        self.fit_kwargs = kwargs

        self.x_axis = None
        self.x_val = None
        self.y_val = None
        self.y_pred = None
        self.prediction_interval = None
        self.resampled_deviation = None

        self._metrics = None
        self._fig = None

    @classmethod
    def _select_data(cls, x: np.ndarray, y: np.ndarray, indices: np.ndarray, fold: int, n_folds: int, shuffle: bool = False):
        n = len(x)
        start_idx = (n // n_folds) * fold
        end_idx = (n // n_folds) * (fold + 1) if fold < n_folds - 1 else n

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
            start_idx = (n // n_folds) * fold
            end_idx = (n // n_folds) * (fold + 1) if fold < n_folds - 1 else n

        train_indices = indices[0: start_idx].copy()
        val_indices = indices[start_idx:end_idx].copy()

        if shuffle:
            np.random.shuffle(val_indices)
            np.random.shuffle(train_indices)

        x_train, y_train, x_val, y_val = x[train_indices], y[train_indices], x[val_indices], y[val_indices]
        return x_train, y_train, x_val, y_val, val_indices

    def _predict(self, x):
        if isinstance(self.model, LinearBootstrap):
            y_pred, prediction_interval, resampled_deviation, *_ = self.model.predict(x=x)
        elif isinstance(self.model, Boosting):
            y_pred, prediction_interval, *_ = self.model.predict(x=x)
            resampled_y = self.model.resample(x=x)
            resampled_deviation = []
            for _y, _y_resampled in zip(y_pred, resampled_y):
                resampled_deviation.append(_y_resampled - _y)
            resampled_deviation = np.array(resampled_deviation)
        else:
            raise NotImplementedError(f'Can not find validation method for {self.model.__class__}')

        return y_pred, prediction_interval, resampled_deviation

    def cross_validate(self, x: np.ndarray, y: np.ndarray):
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

        fold_metrics = {'x_val': [], 'y_val': [], 'y_pred': [], 'index': [], 'prediction_interval': [], 'resampled_deviation': []}

        for fold in range(self.folds):
            if self.strict_no_future:
                x_train, y_train, x_val, y_val, val_indices = self._select_data_sequential(x=x, y=y, indices=indices, fold=fold, n_folds=self.folds, shuffle=self.shuffle)
            else:
                x_train, y_train, x_val, y_val, val_indices = self._select_data(x=x, y=y, indices=indices, fold=fold, n_folds=self.folds, shuffle=self.shuffle)

            self.model.fit(x=x_train, y=y_train, **self.fit_kwargs)

            # Predict on the validation data
            y_pred, prediction_interval, resampled_deviation = self._predict(x_val)

            # this type model will accumulate bootstrap instances on each fit.
            if isinstance(self.model, LinearRegression):
                resampled_deviation = resampled_deviation[:, -self.model.bootstrap_samples:]

            fold_metrics['x_val'].append(x_val)
            fold_metrics['y_val'].append(y_val)
            fold_metrics['y_pred'].append(y_pred)
            fold_metrics['index'].append(val_indices)  # Store sorted indices
            fold_metrics['prediction_interval'].append(prediction_interval)
            fold_metrics['resampled_deviation'].append(resampled_deviation)

        sorted_indices = np.concatenate(fold_metrics['index'])
        for key in ['x_val', 'y_val', 'y_pred', 'prediction_interval', 'resampled_deviation']:
            values = np.concatenate(fold_metrics[key])
            fold_metrics[key] = values[np.argsort(sorted_indices)]

        self.x_val = fold_metrics['x_val']
        self.y_val = fold_metrics['y_val']
        self.y_pred = fold_metrics['y_pred']
        self.prediction_interval = fold_metrics['prediction_interval']
        self.resampled_deviation = fold_metrics['resampled_deviation']

    def validate(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, skip_fitting: bool = False):
        if not skip_fitting:
            self.model.fit(x=x_train, y=y_train, **self.fit_kwargs)

        # Predict on the validation data
        y_pred, prediction_interval, resampled_deviation = self._predict(x_val)

        self.x_val = np.array(x_val)
        self.y_val = np.array(y_val)
        self.y_pred = np.array(y_pred)
        self.prediction_interval = np.array(prediction_interval)
        self.resampled_deviation = np.array(resampled_deviation)

    def plot(self, x_axis: list | np.ndarray = None, **kwargs):
        """
        Plot the validation y_pred and y_val using Plotly.

        Args:
            x_axis (numpy.ndarray, optional): Values for the x-axis. Default is None.

        Returns:
            plotly.graph_objects.Figure: Plotly figure object.
        """
        if self._fig is not None:
            return self._fig

        import plotly.graph_objects as go

        fig = go.Figure()
        y_pred = self.y_pred
        y_val = self.y_val
        prediction_interval = self.prediction_interval

        if x_axis is None:
            if self.x_axis is None:
                x_axis = np.arange(len(y_val))
            else:
                x_axis = self.x_axis

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
                    yaxis='y1'
                )
            )

        # Plot Kelly decision
        if kwargs.get('with_kelly', True):
            kelly_value = []
            for outcome, deviations in zip(self.y_pred, self.resampled_deviation):
                kelly_proportion = kelly_bootstrap(outcomes=np.array(deviations) + outcome, cost=self.trade_cost)
                kelly_value.append(kelly_proportion)

            fig.add_trace(
                go.Bar(
                    x=x_axis,
                    y=np.array(kelly_value),
                    opacity=0.3,
                    name='kelly decision',
                    marker=dict(color='green'),
                    yaxis='y2'
                )
            )

            fig.update_layout(
                yaxis2=dict(
                    title="Kelly",
                    anchor="free",
                    overlaying='y1',
                    tickmode='sync',
                    autoshift=True,
                    showgrid=False,  # Hide grid for y3 axis
                    showline=False,  # Hide line for y3 axis
                    zeroline=False,  # Hide zero line for y3 axis
                    showticklabels=False  # Hide tick labels for y3 axis
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
                range=[-np.ceil(np.quantile(np.abs(y_pred), 0.99) / 0.05) * 0.05, np.ceil(np.quantile(np.abs(y_pred), 0.99) / 0.05) * 0.05],
                dtick=0.05,
                zeroline=True,
                # nticks=int(np.ceil(np.quantile(np.abs(y_pred), 0.95) // 0.05)) * 2 + 1,
                showspikes=True,
                spikesnap='cursor',
            )
        )

        fig.update_xaxes(
            rangebreaks=RANGE_BREAK,
        )

        self._fig = fig
        return fig

    def clear(self):
        self.x_val = None
        self.y_val = None
        self.y_pred = None
        self.prediction_interval = None

        self._metrics = None
        self._fig = None

    @property
    def metrics(self) -> Metrics | None:
        if self.x_val is None or self.y_val is None:
            raise ValueError('Must call .validation() method first')

        if self._metrics is None:
            self._metrics = Metrics(
                model=self.model,
                x=self.x_val,
                y=self.y_val,
                informed_baseline=True,
                cost=self.trade_cost
            )

        return self._metrics
