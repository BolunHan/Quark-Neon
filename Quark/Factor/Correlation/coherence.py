"""
This script defines classes for monitoring and measuring the coherence of price and volume percentage changes
in a stock pool. It utilizes market and trade data from PyQuantKit and includes functionalities for calculating
dispersion coefficients, exponential moving averages (EMA), and regression slopes.

Classes:
- CoherenceMonitor: Monitors and measures the coherence of price percentage change.
- CoherenceEMAMonitor: Extends CoherenceMonitor and includes an EMA for dispersion ratio.
- TradeCoherenceMonitor: Monitors and measures the coherence of volume percentage change based on trade data.

Helper Functions:
- regression(y: list[float] | np.ndarray, x: list[float] | np.ndarray = None) -> float:
    Calculates the slope of linear regression given dependent and independent variables.

Usage:
1. Instantiate the desired monitor class with appropriate parameters.
2. Call the instance with market or trade data to update the monitor.
3. Retrieve the coherence values using the 'value' property of the monitor instance.

Note: This script assumes the availability of AlgoEngine, PyQuantKit, and other required modules.

Author: Bolun
Date: 2023-12-26
"""

import numpy as np
from AlgoEngine.Engine import MarketDataMonitor
from PyQuantKit import MarketData, TradeData

from .. import EMA, MDS, LOGGER


class CoherenceMonitor(MarketDataMonitor):
    """
    Monitors and measures the coherence of price percentage change.

    The correlation coefficient should give a proximate indication of price change coherence
    A factor of up_dispersion / (up_dispersion + down_dispersion) is an indication of (start of) a downward trend

    Attributes:
        update_interval (float): Time interval for updating the monitor.
        sample_interval (float): Time interval for sampling market data.
        weights (dict): Weights for individual stocks in the pool.
        name (str): Name of the monitor.
        monitor_id (str): Identifier for the monitor.
    """

    def __init__(self, update_interval: float, sample_interval: float = 1., weights: dict[str, float] = None, name: str = 'Monitor.Coherence.Price', monitor_id: str = None):
        """
        Initializes the CoherenceMonitor.

        Args:
            update_interval (float): Time interval for updating the monitor.
            sample_interval (float): Time interval for sampling market data.
            weights (dict): Weights for individual stocks in the pool.
            name (str): Name of the monitor.
            monitor_id (str): Identifier for the monitor.
        """
        super().__init__(name=name, monitor_id=monitor_id, mds=MDS)

        self.weights = weights
        self.update_interval = update_interval
        self.sample_interval = sample_interval
        self._historical_price: dict[str, dict[float, float]] = {}
        self._price_change_pct: dict[str, float] = {}
        self._is_ready = True

        # Warning for update_interval
        if update_interval <= 0:
            LOGGER.warning(f'{self.name} should have a positive update_interval')

        # Warning for sample_interval by Shannon's Theorem
        if update_interval / 2 < sample_interval:
            LOGGER.warning(f"{self.name} should have a smaller sample_interval by Shannon's Theorem, max value {update_interval / 2}")

        # Error if sample_interval is not a fraction of update_interval
        if not (update_interval / sample_interval).is_integer():
            LOGGER.error(f"{self.name} should have a smaller sample_interval that is a fraction of the update_interval")

    def __call__(self, market_data: MarketData, **kwargs):
        """
        Updates the monitor with market data.

        Args:
            market_data (MarketData): Market data object containing price information.
        """
        ticker = market_data.ticker
        market_price = market_data.market_price
        timestamp = market_data.timestamp

        self._log_price(ticker=ticker, market_price=market_price, timestamp=timestamp)

        # update price pct change
        baseline_price = list(self._historical_price[ticker].values())[0]  # the sampled_price must be ordered! Guaranteed by consistency of the order of trade data
        price_change_pct = market_price / baseline_price - 1
        self._price_change_pct[ticker] = price_change_pct

    def _log_price(self, ticker: str, market_price: float, timestamp: float):
        """
        Logs sampled prices for historical price tracking.

        Args:
            ticker (str): Ticker symbol of the stock.
            market_price (float): Current market price.
            timestamp (float): Timestamp of the market data.
        """
        sampled_price = self._historical_price.get(ticker, {})

        # update sampled price
        baseline_timestamp = (timestamp // self.update_interval - 1) * self.update_interval
        sample_timestamp = timestamp // self.sample_interval * self.sample_interval
        sampled_price[sample_timestamp] = market_price

        for ts in list(sampled_price):
            if ts < baseline_timestamp:
                sampled_price.pop(ts)
            else:
                break

        self._historical_price[ticker] = sampled_price

    @classmethod
    def regression(cls, y: list[float] | np.ndarray, x: list[float] | np.ndarray = None):
        """
        Calculates the slope of linear regression.

        Args:
            y (list or np.ndarray): Dependent variable.
            x (list or np.ndarray): Independent variable.

        Returns:
            float: Slope of the linear regression.
        """
        y = np.array(y)
        x = np.array(x)

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator:
            slope = numerator / denominator
        else:
            slope = np.nan

        return slope

    def collect_dispersion(self, side: int):
        """
        Collects dispersion data based on price change.

        Args:
            side (int): Sign indicating upward (1) or downward (-1) trend.

        Returns:
            float: Dispersion coefficient.
        """
        price_change_list = []
        weights = []

        # Collect price change data based on weights
        if self.weights:
            for ticker in self.weights:

                if ticker not in self._price_change_pct:
                    continue

                price_change = self._price_change_pct[ticker]

                if price_change * side <= 0:
                    continue

                price_change_list.append(price_change)
                weights.append(self.weights[ticker])

            weights = np.sqrt(weights)

            y = np.array(price_change_list) * weights
            x = (np.argsort(y).argsort() + 1) * weights
        else:
            # If no weights, use default order
            y = np.array(price_change_list)
            x = np.argsort(y).argsort() + 1

        # Check if enough data points for regression
        if len(x) < 3:
            return np.nan

        return self.regression(y=y, x=x)

    def clear(self):
        """Clears historical price and price change data."""
        self._historical_price.clear()
        self._price_change_pct.clear()

    @property
    def value(self) -> dict[str, float]:
        """
        Calculates and returns the dispersion coefficients.

        Returns:
            dict: Dictionary containing 'up', 'down' and 'ratio' dispersion coefficients.
        """
        up_dispersion = self.collect_dispersion(side=1)
        down_dispersion = self.collect_dispersion(side=-1)

        if up_dispersion < 0:
            ratio = 1.
        elif down_dispersion < 0:
            ratio = 0.
        else:
            ratio = down_dispersion / (up_dispersion + down_dispersion)

        return {'up': up_dispersion, 'down': down_dispersion, 'ratio': ratio}

    @property
    def is_ready(self) -> bool:
        """
        Checks if the monitor is ready.

        Returns:
            bool: True if the monitor is ready, False otherwise.
        """
        return self._is_ready


class CoherenceEMAMonitor(CoherenceMonitor, EMA):
    """
    Monitors the Exponential Moving Average (EMA) of the coherence monitor.

    Inherits from CoherenceMonitor and EMA classes.
    """

    def __init__(self, update_interval: float, discount_interval: float, alpha: float, sample_interval: float = 1., weights: dict[str, float] = None, name: str = 'Monitor.Coherence.Price.EMA', monitor_id: str = None):
        """
        Initializes the CoherenceEMAMonitor.

        Args:
            update_interval (float): Time interval for updating the monitor.
            discount_interval (float): Time interval for discounting EMA values.
            alpha (float): Exponential moving average smoothing factor.
            sample_interval (float): Time interval for sampling market data.
            weights (dict): Weights for individual stocks in the pool.
            name (str): Name of the monitor.
            monitor_id (str): Identifier for the monitor.
        """
        super().__init__(update_interval=update_interval, sample_interval=sample_interval, weights=weights, name=name, monitor_id=monitor_id)
        EMA.__init__(self=self, discount_interval=discount_interval, alpha=alpha)

        self.dispersion_ratio = self._register_ema(name='dispersion_ratio')
        self.last_update = 0.

    def __call__(self, market_data: MarketData, **kwargs):
        """
        Updates the CoherenceEMAMonitor with market data.

        Args:
            market_data (MarketData): Market data object containing price information.
        """
        ticker = market_data.ticker
        timestamp = market_data.timestamp

        self._discount_ema(ticker='dispersion_ratio', timestamp=timestamp)
        self._discount_all(timestamp=timestamp)

        super().__call__(market_data=market_data, **kwargs)

        if self.last_update + self.update_interval < timestamp:
            _ = self.value
            self.last_update = timestamp // self.update_interval * self.update_interval

    def clear(self):
        """Clears historical price, price change, and EMA data."""
        super().clear()
        EMA.clear(self)

        self.dispersion_ratio = self._register_ema(name='dispersion_ratio')
        self.last_update = 0.

    @property
    def value(self) -> dict[str, float]:
        """
        Calculates and returns the dispersion coefficients and dispersion ratio.

        Returns:
            dict: Dictionary containing 'up', 'down', and 'ratio' values.
        """
        up_dispersion = self.collect_dispersion(side=1)
        down_dispersion = self.collect_dispersion(side=-1)

        if up_dispersion < 0:
            dispersion_ratio = 1.
        elif down_dispersion < 0:
            dispersion_ratio = 0.
        else:
            dispersion_ratio = down_dispersion / (up_dispersion + down_dispersion)

        self._update_ema(ticker='dispersion_ratio', dispersion_ratio=dispersion_ratio - 0.5)

        return {'up': up_dispersion, 'down': down_dispersion, 'ratio': self.dispersion_ratio.get('dispersion_ratio', np.nan)}


class TradeCoherenceMonitor(CoherenceMonitor):
    """
    Monitors and measures the coherence of volume percentage change based on trade data.

    Inherits from CoherenceMonitor class.
    """

    def __init__(self, update_interval: float, sample_interval: float = 1., weights: dict[str, float] = None, name: str = 'Monitor.Coherence.Volume', monitor_id: str = None):
        """
        Initializes the TradeCoherenceMonitor.

        Args:
            update_interval (float): Time interval for updating the monitor.
            sample_interval (float): Time interval for sampling market data.
            weights (dict): Weights for individual stocks in the pool.
            name (str): Name of the monitor.
            monitor_id (str): Identifier for the monitor.
        """
        super().__init__(
            update_interval=update_interval,
            sample_interval=sample_interval,
            weights=weights,
            name=name,
            monitor_id=monitor_id,
        )

        self._historical_volume: dict[str, dict[float, float]] = {}
        self._historical_volume_net: dict[str, dict[float, float]] = {}

        self._volume_pct: dict[str, float] = {}

    def __call__(self, market_data: MarketData, **kwargs):
        """
        Updates the TradeCoherenceMonitor with market data.

        Args:
            market_data (MarketData): Market data object containing price information.
        """
        super().__call__(market_data=market_data)

        if isinstance(market_data, TradeData):
            self._on_trade(trade_data=market_data)

    def _on_trade(self, trade_data: TradeData):
        """
        Updates volume and net volume based on trade data.

        Args:
            trade_data (TradeData): Trade data object containing volume and side information.
        """
        ticker = trade_data.ticker
        volume = trade_data.volume
        side = trade_data.side.sign
        timestamp = trade_data.timestamp

        sampled_volume = self._historical_volume.get(ticker, {})
        sampled_volume_net = self._historical_volume_net.get(ticker, {})

        # update sampled volume, net volume
        baseline_timestamp = (timestamp // self.update_interval - 1) * self.update_interval
        sample_timestamp = timestamp // self.sample_interval * self.sample_interval
        sampled_volume[sample_timestamp] = sampled_volume.get(sample_timestamp, 0.) + volume
        sampled_volume_net[sample_timestamp] = sampled_volume_net.get(sample_timestamp, 0.) + volume * side

        for ts in list(sampled_volume):
            if ts < baseline_timestamp:
                sampled_volume.pop(ts)
                sampled_volume_net.pop(ts)
            else:
                break

        # update volume percentage change
        baseline_volume = sum(sampled_volume.values())
        net_volume = sum(sampled_volume_net.values())
        self._volume_pct[ticker] = net_volume / baseline_volume

        self._historical_volume[ticker] = sampled_volume
        self._historical_volume_net[ticker] = sampled_volume_net

    @property
    def value(self) -> float:
        """
        Calculates and returns the regression slope between volume percentage change and price percentage change.

        Returns:
            float: the 'slope' value.
        """
        price_pct_change = []
        volume_pct = []

        # Collect price and volume change data based on weights
        if self.weights:
            for ticker in self.weights:
                if ticker in self._price_change_pct:
                    price_pct_change.append(np.sqrt(self.weights[ticker]) * self._price_change_pct[ticker])
                    volume_pct.append(np.sqrt(self.weights[ticker]) * self._volume_pct[ticker])
        else:
            # If no weights, use default order
            price_pct_change.extend(self._price_change_pct.values())
            volume_pct.extend(self._volume_pct.values())

        slope = self.regression(x=volume_pct, y=price_pct_change)

        return slope
