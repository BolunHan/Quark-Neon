import json

import numpy as np

from . import Regression

__all__ = ['LinearRegression', 'RidgeRegression', 'LassoRegression']


class LinearRegression(Regression):
    """
    LinearRegression class for performing linear regression with bootstrapping.

    Note: by default the .predict function will return the mean, not median, of the prediction interval.

    Attributes:
        coefficient (numpy.ndarray): Coefficients of the linear regression model.
        bootstrap_samples (int): Number of bootstrap samples to generate.
        bootstrap_block_size (float): Block size as a percentage of the dataset length for block bootstrap.
        bootstrap_coefficients (list): List to store bootstrap sample coefficients.

    Methods:
        fit(x, y, use_bootstrap=True, method='standard'): Fit the linear regression model to the data.
        bootstrap_standard(x, y): Generate bootstrap samples using the standard method.
        bootstrap_block(x, y): Generate bootstrap samples using the block bootstrap method.
        predict(x, alpha=0.05): Make predictions with prediction intervals.
        plot(x, y, x_axis, alpha=0.05): Plot the data, fitted line, and prediction interval using Plotly.
        to_json(fmt='dict'): Serialize the model to a JSON format.
        from_json(json_str): Deserialize the model from a JSON format.

    Usage:
        model = LinearRegression()
        model.fit(x, y, use_bootstrap=True, method='block')
        model.plot(x=x, y=y, x_axis=index)
    """

    def __init__(self, bootstrap_samples: int = 100, bootstrap_block_size: float = 0.05):
        """
        Initialize the LinearRegression object.

        Args:
            bootstrap_samples (int): Number of bootstrap samples to generate.
            bootstrap_block_size (float): Block size as a percentage of the dataset length for block bootstrap.
        """
        self.coefficient: np.ndarray | None = None

        # parameters for bootstrap
        self.bootstrap_samples = bootstrap_samples
        self.bootstrap_block_size = bootstrap_block_size
        self.bootstrap_coefficients: list[np.ndarray] = []

    def _fit(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coefficient, residuals, *_ = np.linalg.lstsq(x, y, rcond=None)
        return coefficient, residuals

    def fit(self, x: list | np.ndarray, y: list | np.ndarray, use_bootstrap=True, method='standard'):
        """
        Fit the linear regression model to the data.

        Args:
            x (list or numpy.ndarray): Input features.
            y (list or numpy.ndarray): Output values.
            use_bootstrap (bool): Whether to perform bootstrap.
            method (str): Bootstrap method ('standard' or 'block').

        Returns:
            None
        """
        coefficient, residuals = self._fit(x=x, y=y)

        if use_bootstrap:
            if method == 'standard':
                self.bootstrap_standard(x, y)
            elif method == 'block':
                self.bootstrap_block(x, y)
            else:
                raise ValueError("Invalid bootstrap method. Use 'standard' or 'block'.")

        self.coefficient = coefficient

        return coefficient, residuals

    def bootstrap_standard(self, x, y):
        """
        Generate bootstrap samples using the standard method.

        Args:
            x (list or numpy.ndarray): Input features.
            y (list or numpy.ndarray): Output values.

        Returns:
            None
        """
        n = len(x)
        for _ in range(self.bootstrap_samples):
            indices = np.random.choice(n, n, replace=True)
            x_sampled, y_sampled = x[indices], y[indices]
            coefficient, _ = self._fit(x=x_sampled, y=y_sampled)
            self.bootstrap_coefficients.append(coefficient)

    def bootstrap_block(self, x, y):
        """
        Generate bootstrap samples using the block bootstrap method.

        Args:
            x (list or numpy.ndarray): Input features.
            y (list or numpy.ndarray): Output values.

        Returns:
            None
        """
        n = len(x)
        block_size = int(self.bootstrap_block_size * n)
        num_blocks = n // block_size

        for _ in range(self.bootstrap_samples):
            indices = []
            for _ in range(num_blocks):
                block_start = np.random.randint(0, n - block_size + 1)
                indices.extend(range(block_start, block_start + block_size))

            x_sampled, y_sampled = x[indices], y[indices]
            coefficient, _ = self._fit(x=x_sampled, y=y_sampled)
            self.bootstrap_coefficients.append(coefficient)

    def predict(self, x: list | np.ndarray, alpha=0.05):
        """
        Make predictions with prediction intervals.

        Args:
            x (list or numpy.ndarray): Input features.
            alpha (float): Significance level for the prediction interval.

        Returns:
            Tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray): Predicted values, prediction interval, and bootstrap residuals.
        """
        x = np.array(x)

        # Compute mean predictions and intervals for all input sets
        y_pred = np.dot(x, self.coefficient)

        if not self.bootstrap_coefficients:
            return y_pred, None, None, np.nan

        bootstrap_deviation = []
        for bootstrap_coefficient in self.bootstrap_coefficients:
            y_bootstrap = np.dot(x, bootstrap_coefficient)
            deviation = y_bootstrap - y_pred
            bootstrap_deviation.append(deviation)

        bootstrap_deviation = np.array(bootstrap_deviation).T

        # For single input, X might be a 1D array
        if len(x.shape) == 1:
            lower_bound = np.quantile(bootstrap_deviation, alpha / 2)
            upper_bound = np.quantile(bootstrap_deviation, 1 - alpha / 2)
            variance = np.var(bootstrap_deviation)
        else:
            lower_bound = np.quantile(bootstrap_deviation, alpha / 2, axis=1)
            upper_bound = np.quantile(bootstrap_deviation, 1 - alpha / 2, axis=1)
            variance = np.var(bootstrap_deviation, axis=1, ddof=1)

        interval = np.array([lower_bound, upper_bound]).T

        return y_pred, interval, bootstrap_deviation, variance

    def plot(self, x, y, x_axis, alpha=0.05, **kwargs):
        """
        Plot the data, fitted line, and prediction interval using Plotly.

        Args:
            x (list or numpy.ndarray): Input features.
            y (list or numpy.ndarray): Output values.
            x_axis (list or numpy.ndarray): Values for the (plotted) x-axis.
            alpha (float): Significance level for the prediction interval.
            **kwargs: Additional keyword arguments for customizing the plot:
                - 'data_name' (str, optional): Name for the data trace on the plot. Default is "Data".
                - 'model_name' (str, optional): Name for the fitted line trace on the plot. Default is "Fitted Line".
                - 'title' (str, optional): Title for the plot. Default is "Bootstrap Linear Regression".
                - 'x_name' (str, optional): Label for the x-axis. Default is "Index".
                - 'y_name' (str, optional): Label for the y-axis. Default is "Y".

        Returns:
            plotly.graph_objects.Figure: Plotly figure object.
        """
        import plotly.graph_objects as go

        y_pred, interval, _, variance = self.predict(x, alpha)

        fig = go.Figure()

        # Scatter plot for data
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=y,
                mode='markers',
                name=kwargs.get('data_name', "Data"),
                yaxis='y1'
            )
        )

        # Line plot for fitted line
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=y_pred,
                mode='lines',
                name=kwargs.get('model_name', "Fitted Line"),
                line=dict(color='red'),
                yaxis='y1'
            )
        )

        # Fill the area between the prediction intervals
        if self.bootstrap_coefficients:
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

            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=np.sqrt(variance),
                    mode='lines',
                    name="Estimated Variance",
                    line=dict(color='blue'),
                    yaxis='y2'  # Use the secondary y-axis
                )
            )

        # Layout settings
        fig.update_layout(
            title=kwargs.get('title', "Bootstrap Linear Regression"),
            xaxis_title=kwargs.get('x_name', "Index"),
            yaxis_title=kwargs.get('y_name', "Y"),
            hovermode="x unified",  # Enable hover for the x-axis
            template='simple_white',
            yaxis=dict(
                showspikes=True
            ),
            yaxis2=dict(
                title="Variance",
                overlaying='y',
                side='right',
                showgrid=False
            )
        )

        return fig

    def to_json(self, fmt='dict') -> dict | str:
        """
        Serialize the model to a JSON format.

        Args:
            fmt (str): Format for serialization ('dict' or 'json').

        Returns:
            dict or str: Serialized model.
        """
        json_dict = dict(
            coefficient=self.coefficient.tolist() if self.coefficient is not None else None,
            bootstrap_samples=self.bootstrap_samples,
            bootstrap_block_size=self.bootstrap_block_size,
            bootstrap_coefficients=[_.tolist() for _ in self.bootstrap_coefficients]
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

        self = cls(
            bootstrap_samples=json_dict['bootstrap_samples'],
            bootstrap_block_size=json_dict['bootstrap_block_size']
        )

        self.coefficient = np.array(json_dict['coefficient'])
        self.bootstrap_coefficients.extend([np.array(_) for _ in json_dict['bootstrap_coefficients']])

        return self


class RidgeRegression(LinearRegression):
    """
    BootstrapRidgeRegression class for performing ridge regression with bootstrapping.

    Attributes:
        coefficient (numpy.ndarray): Coefficients of the ridge regression model.
        bootstrap_samples (int): Number of bootstrap samples to generate.
        bootstrap_block_size (float): Block size as a percentage of the dataset length for block bootstrap.
        bootstrap_coefficients (list): List to store bootstrap sample coefficients.
        alpha (float): Regularization strength.

    Methods:
        fit(x, y, use_bootstrap=True, method='standard'): Fit the ridge regression model to the data.
        bootstrap_standard(x, y): Generate bootstrap samples using the standard method.
        bootstrap_block(x, y): Generate bootstrap samples using the block bootstrap method.
    """

    def __init__(self, bootstrap_samples: int = 100, bootstrap_block_size: float = 0.05, alpha: float = 1.0):
        """
        Initialize the BootstrapRidgeRegression object.

        Args:
            bootstrap_samples (int): Number of bootstrap samples to generate.
            bootstrap_block_size (float): Block size as a percentage of the dataset length for block bootstrap.
            alpha (float): Regularization strength.
        """
        super().__init__(bootstrap_samples=bootstrap_samples, bootstrap_block_size=bootstrap_block_size)
        self.alpha = alpha

    def _fit(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, m = x.shape
        identity_matrix = np.eye(m)
        regularization_matrix = self.alpha * identity_matrix

        x_transpose_x = np.dot(x.T, x)
        ridge_matrix = x_transpose_x + regularization_matrix
        ridge_matrix_inv = np.linalg.inv(ridge_matrix)
        coefficient = np.dot(np.dot(ridge_matrix_inv, x.T), y)

        residuals = y - np.dot(x, coefficient)

        return coefficient, residuals


class LassoRegression(LinearRegression):
    def __init__(self, bootstrap_samples: int = 100, bootstrap_block_size: float = 0.05, alpha: float = 1.0):
        super().__init__(bootstrap_samples, bootstrap_block_size)
        self.alpha = alpha

        self.max_iter = 1000
        self.tolerance = 1e-4

    def _fit(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.linear_model import Lasso
        # Create Lasso regression model
        lasso_model = Lasso(alpha=self.alpha)

        # Fit the model
        lasso_model.fit(x, y)

        # Extract coefficients and residuals
        coefficients = lasso_model.coef_
        residuals = y - lasso_model.predict(x)

        return coefficients, residuals

    def _fit(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, m = x.shape

        # Use coordinate descent to solve Lasso regression
        # This is a basic implementation; you might want to use a specialized library for better performance

        w = np.zeros(m)

        for _ in range(self.max_iter):
            w_old = np.copy(w)

            for j in range(m):
                x_j = x[:, j]
                residual = y - np.dot(x, w) + w[j] * x_j
                rho = np.dot(x_j, residual)
                z = np.dot(x_j, x_j)

                # Soft thresholding
                w[j] = np.sign(rho) * max(0, abs(rho) - self.alpha) / z if z != 0 else 0

            if np.sum(np.abs(w - w_old)) < self.tolerance:
                break

        residuals = y - np.dot(x, w)

        return w, residuals


def test():
    """
    Test the LinearRegression class with synthetic data.

    Returns:
        None
    """
    np.random.seed(42)
    n = 100
    index = np.arange(n)
    x = np.column_stack((np.ones(n), index, 3 * index + 20 + 0.04 * (index - 50) ** 2))
    y = 5 * index + 5 * (index - 50) ** 2 + np.random.normal(scale=500, size=n) + np.random.normal(scale=100, size=n) * (index - 50)

    # Example usage:
    model = LinearRegression(bootstrap_samples=50)
    model.fit(x, y, use_bootstrap=True, method='block')

    json_dump = model.to_json()
    model = LinearRegression.from_json(json_dump)

    # Use the index as the x_axis
    fig = model.plot(x=x, y=y, x_axis=index)
    fig.show()


if __name__ == '__main__':
    test()
