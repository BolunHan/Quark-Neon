import numpy as np


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

        self.metrics: dict[str, float | np.ndarray | None] = {'mse': 0., 'residuals': None, 'y_actual': None, 'y_pred': None, 'prediction_interval': None}

    @classmethod
    def _compute_metrics(cls, y_actual, y_pred, prediction_interval):
        """
        Compute metrics (MSE, residuals, and prediction interval) for validation.

        Args:
            y_actual (numpy.ndarray): Actual output values.
            y_pred (numpy.ndarray): Predicted output values.
            prediction_interval (numpy.ndarray): Prediction interval.

        Returns:
            Tuple[float, numpy.ndarray, numpy.ndarray]: MSE, residuals, and prediction interval.
        """
        residuals = y_actual - y_pred
        mse = np.mean(residuals ** 2)
        return mse, residuals, prediction_interval

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

        fold_metrics = {'mse': [], 'residuals': [], 'y_actual': [], 'y_pred': [], 'index': [], 'prediction_interval': []}

        for fold in range(self.folds):
            if self.strict_no_future:
                x_train, y_train, x_val, y_val, val_indices = self._select_data_sequential(x=x, y=y, indices=indices, fold=fold, n_folds=self.folds, shuffle=self.shuffle)
            else:
                x_train, y_train, x_val, y_val, val_indices = self._select_data(x=x, y=y, indices=indices, fold=fold, n_folds=self.folds, shuffle=self.shuffle)

            self.model.fit(x=x_train, y=y_train, **self.fit_kwargs)

            # Predict on the validation data
            y_pred, prediction_interval, *_ = self.model.predict(x_val)

            # Compute metrics for the fold
            mse, residuals, _ = self._compute_metrics(y_val, y_pred, prediction_interval)

            fold_metrics['mse'].append(mse)
            fold_metrics['residuals'].append(residuals)
            fold_metrics['y_actual'].append(y_val)
            fold_metrics['y_pred'].append(y_pred)
            fold_metrics['index'].append(val_indices)  # Store sorted indices
            fold_metrics['prediction_interval'].append(prediction_interval)

        sorted_indices = np.concatenate(fold_metrics['index'])
        for key in ['residuals', 'y_actual', 'y_pred', 'prediction_interval']:
            values = np.concatenate(fold_metrics[key])
            fold_metrics[key] = values[np.argsort(sorted_indices)]

        # Store average metrics across folds
        self.metrics['mse'] = np.mean(fold_metrics['mse'], axis=0)
        self.metrics['residuals'] = fold_metrics['residuals']
        self.metrics['y_actual'] = fold_metrics['y_actual']
        self.metrics['y_pred'] = fold_metrics['y_pred']
        self.metrics['prediction_interval'] = fold_metrics['prediction_interval']

    def validate_out_sample(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray):
        self.model.fit(x=x_train, y=y_train, **self.fit_kwargs)

        # Predict on the validation data
        y_pred, prediction_interval, *_ = self.model.predict(x_val)

        # Compute metrics for the fold
        mse, residuals, _ = self._compute_metrics(y_val, y_pred, prediction_interval)

        self.metrics['mse'] = mse
        self.metrics['residuals'] = np.array(residuals)
        self.metrics['y_actual'] = np.array(y_val)
        self.metrics['y_pred'] = np.array(y_pred)
        self.metrics['prediction_interval'] = np.array(prediction_interval)

    def _compute_accuracy(self, y_actual, y_pred):
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

    def _compute_significant_accuracy(self, y_actual, y_pred, prediction_interval):
        """
        Compute accuracy for significant predictions based on the prediction interval.

        Args:
            y_actual (numpy.ndarray): Actual output values.
            y_pred (numpy.ndarray): Predicted output values.
            prediction_interval (numpy.ndarray): Prediction interval.

        Returns:
            float: Significant accuracy.
        """
        lower_bound_sign = y_pred + prediction_interval[:, 0]
        upper_bound_sign = y_pred + prediction_interval[:, 1]

        correct_signs = (y_actual > 0) & (lower_bound_sign > 0) | (y_actual < 0) & (upper_bound_sign < 0)
        significant_accuracy = np.mean(correct_signs)
        return significant_accuracy

    def _compute_roc(self, y_actual, bootstrap_residuals, alpha_range=np.linspace(0.01, 0.5, 50)):
        """
        Compute Receiver Operating Characteristic (RoC) curve.

        Args:
            y_actual (numpy.ndarray): Actual output values.
            bootstrap_residuals (numpy.ndarray): Bootstrap residuals.
            alpha_range (numpy.ndarray): Range of significance thresholds (alpha values).

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: Alpha values and corresponding RoC values.
        """
        roc_values = []
        for alpha in alpha_range:
            lower_bound = np.quantile(bootstrap_residuals, alpha / 2)
            upper_bound = np.quantile(bootstrap_residuals, 1 - alpha / 2)
            correct_signs = (y_actual > 0) & (lower_bound > 0) | (y_actual < 0) & (upper_bound < 0)
            roc_values.append(np.mean(correct_signs))

        return alpha_range, np.array(roc_values)

    def plot(self, x_axis: list | np.ndarray = None, **kwargs):
        """
        Plot the validation y_pred and y_actual using Plotly.

        Args:
            x_axis (numpy.ndarray, optional): Values for the x-axis. Default is None.

        Returns:
            plotly.graph_objects.Figure: Plotly figure object.
        """
        import plotly.graph_objects as go

        fig = go.Figure()
        y_pred = self.metrics['y_pred']
        y_actual = self.metrics['y_actual']
        interval = self.metrics['prediction_interval']

        if x_axis is None:
            x_axis = np.arange(len(y_actual))

        # Scatter plot for actual values
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=y_actual,
                mode='markers',
                name=kwargs.get('data_name', "y_actual"),
                yaxis='y1'
            )
        )

        # Line plot for predicted values
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=y_pred,
                mode='lines',
                name=kwargs.get('model_name', "y_val"),
                line=dict(color='red'),
                yaxis='y1'
            )
        )

        # Plot prediction interval (shadow area)
        if 'prediction_interval' in self.metrics and len(self.metrics['prediction_interval']) > 0:
            fig.add_trace(
                go.Scatter(
                    name=f'Upper Bound',
                    x=x_axis,
                    y=y_pred + interval[:, 1],
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
                    y=y_pred + interval[:, 0],
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
