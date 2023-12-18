from AlgoEngine.Engine import MarketDataMonitor
from PyQuantKit import MarketData, TradeData

from .. import EMA, Synthetic, MDS, DEBUG_MODE


class AggressivenessMonitor(MarketDataMonitor):
    """
    monitor the aggressiveness of buy / sell trades. the deeper the orders fill at one time, the more aggressive they are

    this module requires 2 additional fields
    - sell_order_id
    - buy_order_id

    without these 2 fields the monitor is set to is_ready = False
    the monitor aggregated aggressive buying / selling volume separately
    """

    def __init__(self, name: str = 'Monitor.Aggressiveness', monitor_id: str = None):
        super().__init__(
            name=name,
            monitor_id=monitor_id,
            mds=MDS
        )

        self._last_update: dict[str, float] = dict()
        self._trade_price: dict[str, dict[int, float]] = dict()
        self._aggressive_buy: dict[str, float] = {}
        self._aggressive_sell: dict[str, float] = {}
        self._is_ready = True

    def __call__(self, market_data: MarketData, **kwargs):
        if isinstance(market_data, TradeData):
            self._on_trade(trade_data=market_data)

    def _on_trade(self, trade_data: TradeData):
        ticker = trade_data.ticker
        price = trade_data.market_price
        volume = trade_data.volume
        side = trade_data.side.sign
        timestamp = trade_data.timestamp

        if ticker in self._trade_price:
            trade_price_log = self._trade_price[ticker]
        else:
            trade_price_log = self._trade_price[ticker] = {}

        # trade logs triggered by the same order can not have 2 different timestamp
        if ticker not in self._last_update or self._last_update[ticker] < timestamp:
            trade_price_log.clear()
            self._last_update[ticker] = timestamp
            return

        if side > 0:  # a buy init trade
            if 'buy_order_id' not in trade_data.additional:
                self._is_ready = False

            order_id = trade_data.additional['buy_order_id']
        elif side < 0:
            if 'sell_order_id' not in trade_data.additional:
                self._is_ready = False

            order_id = trade_data.additional['sell_order_id']
        else:
            return

        if order_id not in trade_price_log:
            trade_price_log[order_id] = price
        else:
            last_price = trade_price_log[order_id]

            if last_price == price:
                pass
            else:
                self._update_aggressiveness(
                    ticker=ticker,
                    side=side,
                    volume=volume,
                    timestamp=timestamp
                )

        self._last_update[ticker] = timestamp

    def _update_aggressiveness(self, ticker: str, volume: float, side: int, timestamp: float):
        if side > 0:
            self._aggressive_buy[ticker] = self._aggressive_buy.get(ticker, 0.) + volume
        else:
            self._aggressive_sell[ticker] = self._aggressive_sell.get(ticker, 0.) + volume

    @property
    def value(self) -> tuple[dict[str, float], dict[str, float]]:
        return self._aggressive_buy, self._aggressive_sell

    @property
    def is_ready(self) -> bool:
        return self._is_ready


class AggressivenessEMAMonitor(AggressivenessMonitor, EMA, Synthetic):
    """
    ema average of aggressiveness buying / selling volume.

    Note that the discount process is triggered by on_trade_data.
    - Upon discount, there should be a discontinuity of index value.
    - But if the update interval is small enough, the effect should be marginal

    aggressiveness is a relative stable indication. a large positive aggressiveness indicates market is in an upward ongoing trend
    """

    def __init__(self, discount_interval: float, alpha: float, weights: dict[str, float], normalized: bool = True, name: str = 'Monitor.Aggressiveness.EMA', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id)
        EMA.__init__(self=self, discount_interval=discount_interval, alpha=alpha)
        Synthetic.__init__(self=self, weights=weights)

        self.normalized = normalized

        self._aggressive_buy: dict[str, float] = self._register_ema(name='aggressive_buy')  # ema of the aggressiveness buy volume
        self._aggressive_sell: dict[str, float] = self._register_ema(name='aggressive_sell')  # ema of the aggressiveness sell volume
        self._trade_volume: dict[str, float] = self._register_ema(name='trade_volume')  # ema of the total volume

    def _update_aggressiveness(self, ticker: str, volume: float, side: int, timestamp: float):

        if side > 0:
            self._accumulate_ema(ticker=ticker, timestamp=timestamp, replace_na=0., aggressive_buy=volume, aggressive_sell=0.)
        else:
            self._accumulate_ema(ticker=ticker, timestamp=timestamp, replace_na=0., aggressive_buy=0., aggressive_sell=volume)

        if DEBUG_MODE:
            total_volume = self._trade_volume[ticker]
            aggressiveness_volume = self._aggressive_buy[ticker] if side > 0 else self._aggressive_sell[ticker]
            adjusted_volume_ratio = aggressiveness_volume / total_volume if total_volume else 0.

            if not (0 <= adjusted_volume_ratio <= 1):
                raise ValueError(f'{ticker} {self.__class__.__name__} encounter invalid value, total_volume={total_volume}, aggressiveness={aggressiveness_volume}, side={side}')

    def _on_trade(self, trade_data: TradeData):
        ticker = trade_data.ticker
        self._accumulate_ema(ticker=ticker, trade_volume=trade_data.volume, replace_na=0.)  # to avoid redundant calculation, timestamp is not passed-in, so that the discount function will not be triggered
        super()._on_trade(trade_data=trade_data)

    def __call__(self, market_data: MarketData, **kwargs):
        ticker = market_data.ticker
        timestamp = market_data.timestamp

        self._discount_ema(ticker=ticker, timestamp=timestamp)
        self._discount_all(timestamp=timestamp)

        return super().__call__(market_data=market_data, **kwargs)

    @property
    def value(self) -> tuple[dict[str, float], dict[str, float]]:
        aggressive_buy = {}
        aggressive_sell = {}

        for ticker, volume in self._trade_volume.items():
            if volume:
                aggressive_buy[ticker] = self._aggressive_buy.get(ticker, 0.) / volume
                aggressive_sell[ticker] = self._aggressive_sell.get(ticker, 0.) / volume

        return aggressive_buy, aggressive_sell

    @property
    def index_value(self) -> float:
        aggressive_buy, aggressive_sell = self.value
        values = {}

        for ticker in set(aggressive_buy) | set(aggressive_sell):
            values[ticker] = aggressive_buy.get(ticker, 0) - aggressive_sell.get(ticker, 0.)

        return self.composite(values)
