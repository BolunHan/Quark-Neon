import numpy as np


class CrossValidation:
    def __init__(self, model, folds=5, shuffle=True, **kwargs):
        """
        Initialize the CrossValidation object.

        Args:
            model: Regression model object (e.g., BootstrapLinearRegression).
            folds (int): Number of folds for cross-validation.
            shuffle (bool): Whether to use random data split or not.
        """
        self.model = model
        self.folds = folds
        self.shuffle = shuffle

        self.fit_kwargs = kwargs

        self.metrics = {'mse': [], 'residuals': [], 'y_actual': [], 'y_val': [], 'prediction_interval': []}

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

        if self.shuffle:
            np.random.shuffle(indices)

        fold_metrics = {'mse': [], 'residuals': [], 'y_actual': [], 'y_pred': [], 'index': [], 'prediction_interval': []}

        for fold in range(self.folds):
            start_idx = n // self.folds * fold
            end_idx = n // self.folds * (fold + 1) if fold < self.folds - 1 else n
            val_indices = indices[start_idx:end_idx]
            train_indices = np.setdiff1d(indices, val_indices)
            x_train, y_train, x_val, y_val = x[train_indices], y[train_indices], x[val_indices], y[val_indices]

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
        self.metrics['y_val'] = fold_metrics['y_pred']
        self.metrics['prediction_interval'] = fold_metrics['prediction_interval']

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
        y_pred = self.metrics['y_val']
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
