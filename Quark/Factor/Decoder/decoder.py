"""
This script defines classes for monitoring and decoding market data into different trends.
It includes a DecoderMonitor class and an IndexDecoderMonitor class, both utilizing the OnlineDecoder
and Synthetic classes.

Classes:
- DecoderMonitor: Monitors market data and decodes it into different trends based on specified thresholds.
- IndexDecoderMonitor: Extends DecoderMonitor and includes a Synthetic index for monitoring.

Usage:
1. Instantiate the desired monitor class with appropriate parameters.
2. Call the instance with market data to update the monitor.
3. Retrieve the decoding results using the 'value' property of the monitor instance.
4. Register some callback function for wavelet confirmation is recommended.

Note: This script assumes the availability of AlgoEngine, PyQuantKit, and other required modules.

Author: Bolun
Date: 2023-12-26
"""

from PyQuantKit import MarketData

from .. import Synthetic, FactorMonitor
from ..decoder import OnlineDecoder, Wavelet


class DecoderMonitor(FactorMonitor, OnlineDecoder):
    """
    Monitors and decodes market data into different trends.

    Marks the market movement into different trends:
    - When price goes up to 1% (up_threshold).
    - When price goes down to 1% (down_threshold).
    - When price goes up/down relative to the local minimum/maximum more than 0.5% (confirmation_level).
    - When the market trend goes on for more than 15 * 60 seconds (timeout).

    Upon a new trend being confirmed, a callback function will be called.

    Attributes:
        confirmation_level (float): Confirmation level for trend detection.
        timeout (float): Timeout duration for a trend.
        up_threshold (float): Upward price movement threshold.
        down_threshold (float): Downward price movement threshold.
        retrospective (bool): Whether to apply the decoder retrospectively.
        name (str): Name of the monitor.
        monitor_id (str): Identifier for the monitor.
    """

    def __init__(self, confirmation_level: float = 0.005, timeout: float = 15 * 60, up_threshold: float = 0.01, down_threshold: float = 0.01, retrospective: bool = False, name: str = 'Monitor.Decoder', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id)
        OnlineDecoder.__init__(self=self, confirmation_level=confirmation_level, timeout=timeout, up_threshold=up_threshold, down_threshold=down_threshold, retrospective=retrospective)

    def __call__(self, market_data: MarketData, **kwargs):
        """
        Updates the DecoderMonitor with market data.

        Args:
            market_data (MarketData): Market data object containing price information.
        """
        ticker = market_data.ticker
        market_price = market_data.market_price
        timestamp = market_data.timestamp

        self.update_decoder(ticker=ticker, market_price=market_price, timestamp=timestamp)

    def clear(self):
        """
        Clears the historical price, price change, and decoder state history.
        """
        OnlineDecoder.clear(self)

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}.{ticker}' for ticker in subscription
        ]

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        raise NotImplementedError()

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict):
        raise NotImplementedError()

    @property
    def value(self) -> dict[str, list[Wavelet]]:
        """
        Gets the state history of the decoder.

        Returns:
            dict[str, list[Wavelet]]: State history of the decoder.
        """
        return self.state_history

    @property
    def is_ready(self) -> bool:
        """
        Checks if the DecoderMonitor is ready.

        Returns:
            bool: True if the monitor is ready, False otherwise.
        """
        return False


class IndexDecoderMonitor(DecoderMonitor, Synthetic):
    """
    Monitors and decodes market data into different trends for a synthetic index.

    Inherits from DecoderMonitor and Synthetic classes.

    Attributes:
        weights (dict[str, float]): Weights for individual stocks in the synthetic index.
    """

    def __init__(self, weights: dict[str, float], confirmation_level: float = 0.005, timeout: float = 15 * 60, up_threshold: float = 0.01, down_threshold: float = 0.01, retrospective: bool = False, name: str = 'Monitor.Decoder.Index', monitor_id: str = None):
        super().__init__(confirmation_level=confirmation_level, timeout=timeout, up_threshold=up_threshold, down_threshold=down_threshold, retrospective=retrospective, name=name, monitor_id=monitor_id)
        Synthetic.__init__(self=self, weights=weights)

    def __call__(self, market_data: MarketData, **kwargs):
        """
        Updates the IndexDecoderMonitor with market data.

        Args:
            market_data (MarketData): Market data object containing price information.
        """
        self.update_synthetic(ticker=market_data.ticker, market_price=market_data.market_price)
        self.update_decoder(ticker='synthetic', market_price=self.synthetic_index, timestamp=market_data.timestamp)

    def clear(self):
        """
        Clears the historical price, price change, decoder state history, and synthetic data.
        """
        super().clear()
        Synthetic.clear(self)

    @property
    def value(self) -> list[Wavelet]:
        """
        Gets the state history of the synthetic index.

        Returns:
            list[Wavelet]: State history of the synthetic index.
        """
        return self.state_history['synthetic']
