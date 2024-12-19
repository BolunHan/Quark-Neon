import abc
import datetime
import time
from typing import Literal

import numpy as np
from algo_engine.backtest import SimMatch, ProgressiveReplay
from algo_engine.backtest.metrics import TradeMetrics
from algo_engine.base import MarketData, TradeReport, TradeInstruction
from algo_engine.profile import Profile, PROFILE

from . import LOGGER
from .web_app import WebApp


class Tester(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            start_date: datetime.date,
            end_date: datetime.date,
            dtype: list[str] = None,
            profile: Profile = None,
            **kwargs
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.dtype = ['TickData', 'TradeData'] if dtype is None else dtype
        self.profile = PROFILE if profile is None else profile

        self.timestamp = 0.
        self.last_price = {}
        self.subscription = set()
        self.web_app = WebApp(start_date=start_date, end_date=end_date, **kwargs)
        self.metrics: dict[str, TradeMetrics] = {}
        self.sim_match: dict[str, SimMatch] = {}

    def register_ticker(self, ticker: str, **kwargs):
        self.subscription.add(ticker)

        self.metrics[ticker] = TradeMetrics()

        self.web_app.register(ticker=ticker, **kwargs)

        sim_match = self.sim_match[ticker] = SimMatch(
            ticker=ticker,
            instant_fill=kwargs.get('instant_fill', True)
        )

        # to add callback function to sim_match, use following codes.
        # sim_match.on_order = self.on_order
        # sim_match.on_report = self.on_report

    def unregister_ticker(self, ticker: str, **kwargs):
        self.subscription.remove(ticker)

        self.metrics.pop(ticker)

        # the web app does not provide an unregister method, however, this is not a requirement

        sim_match = self.sim_match.pop(ticker)
        sim_match.unregister()

    def _launch_order(self, ticker: str, volume: float, limit_price: float):
        order = TradeInstruction(ticker=ticker, side=np.sign(volume), volume=abs(float), timestamp=self.timestamp)
        self.sim_match[ticker].launch_order(order=order)

    def buy(self, ticker: str, volume: float = None, limit_price: float = None):
        if volume is None:
            trade_metrics = self.metrics[ticker]
            exposure = trade_metrics.exposure
            volume = -exposure if exposure < 0 else 1

        if limit_price is None:
            limit_price = self.last_price[ticker]

        self._launch_order(ticker=ticker, volume=volume, limit_price=limit_price)

    def sell(self, ticker: str, volume: float = None, limit_price: float = None):
        if volume is None:
            trade_metrics = self.metrics[ticker]
            exposure = trade_metrics.exposure
            volume = -exposure if exposure > 0 else -1

        if limit_price is None:
            limit_price = self.last_price[ticker]

        self._launch_order(ticker=ticker, volume=volume, limit_price=limit_price)

    @abc.abstractmethod
    def load_data(self, ticker: str, market_date: datetime.date, dtype: Literal['TickData', 'TradeData', 'TransactionData', 'OrderBook']) -> list[MarketData]:
        ...

    @abc.abstractmethod
    def on_market_data(self, market_data: MarketData, **kwargs):
        ...

    @abc.abstractmethod
    def on_report(self, report: TradeReport, **kwargs):
        ...

    @abc.abstractmethod
    def on_order(self, order: TradeInstruction, **kwargs):
        ...

    def bod(self, market_date: datetime.date, **kwargs):
        pass

    def eod(self, market_date: datetime.date, **kwargs):
        pass

    def run(self, **kwargs):
        replay = ProgressiveReplay(
            loader=self.load_data,
            tickers=list(self.subscription),
            dtype=['TickData', 'TradeData'],
            start_date=self.start_date,
            end_date=self.end_date,
            bod=self.bod,
            eod=self.eod,
            tick_size=kwargs.get('progress_tick_size', 0.001),
        )

        _start_ts = time.time()

        for market_data in replay:
            self.on_market_data(market_data=market_data)
            self.sim_match[market_data.ticker](market_data=market_data)
            self.web_app.update(market_data=market_data)

            self.timestamp = market_data.timestamp
            self.last_price[market_data.ticker] = market_data.market_price

        LOGGER.info(f'All done! time_cost: {time.time() - _start_ts:,.3}s')


class StrategyTester(Tester):
    from algo_engine.strategy.strategy_engine import StrategyEngine

    def __init__(self, start_date: datetime.date, end_date: datetime.date, data_loader, strategy: StrategyEngine, **kwargs):
        self.data_loader = data_loader
        self.strategy = strategy
        self.event_engine = self.strategy.event_engine
        self.topic_set = self.strategy.topic_set
        self.multi_threading = kwargs.get('multi_threading', False)
        self.lock = self.strategy.lock

        super().__init__(
            start_date=start_date,
            end_date=end_date,
            dtype=kwargs.pop('dtype', ['TickData', 'TradeData']),
            profile=kwargs.pop('profile', PROFILE),
            event_engine=strategy.event_engine,
            topic_set=strategy.topic_set,
            multi_threading=kwargs.pop('multi_threading', False),
            **kwargs
        )

    def register_ticker(self, ticker: str, **kwargs):
        super().register_ticker(ticker=ticker, **kwargs)

        for ticker, sim_match in self.sim_match.items():
            sim_match.register(event_engine=self.event_engine, topic_set=self.topic_set)

    def register(self):
        self.event_engine.register_handler(topic=self.topic_set.realtime, handler=self.strategy.mds.on_market_data)
        self.event_engine.register_handler(topic=self.topic_set.realtime, handler=self.strategy.position_tracker.on_market_data)
        self.event_engine.register_handler(topic=self.topic_set.realtime, handler=self.on_market_data)

        self.event_engine.register_handler(topic=self.topic_set.on_order, handler=self.strategy.balance.on_order)
        self.event_engine.register_handler(topic=self.topic_set.on_order, handler=self.on_order)
        self.event_engine.register_handler(topic=self.topic_set.on_report, handler=self.strategy.balance.on_report)
        self.event_engine.register_handler(topic=self.topic_set.on_report, handler=self.on_report)

    def initialize_position_management(self):
        for ticker in self.subscription:
            risk_profile = self.strategy.position_tracker.dma.risk_profile

            risk_profile.set_rule(ticker=ticker, key='max_trade_long', value=np.inf)
            risk_profile.set_rule(ticker=ticker, key='max_trade_short', value=np.inf)
            risk_profile.set_rule(ticker=ticker, key='max_exposure_long', value=np.inf)
            risk_profile.set_rule(ticker=ticker, key='max_exposure_short', value=np.inf)

    def load_data(self, ticker: str, market_date: datetime.date, dtype: Literal['TickData', 'TradeData', 'TransactionData', 'OrderBook']) -> list[MarketData]:
        return self.data_loader(ticker=ticker, market_date=market_date, dtype=dtype)

    def bod(self, market_date: datetime.date, **kwargs):
        super().bod(market_date=market_date, **kwargs)
        self.bod(market_date=market_date, **kwargs)

    def eod(self, market_date: datetime.date, **kwargs):
        super().bod(market_date=market_date, **kwargs)
        self.bod(market_date=market_date, **kwargs)

    def on_market_data(self, market_data: MarketData, **kwargs):
        self.strategy.__call__(market_data=market_data, **kwargs)

        if self.lock.locked():
            self.lock.release()

    def on_report(self, report: TradeReport, **kwargs):
        self.strategy.on_report(report=report, **kwargs)

    def on_order(self, order: TradeInstruction, **kwargs):
        self.strategy.on_order(order=order, **kwargs)

    def _launch_order(self, ticker: str, volume: float, limit_price: float):
        self.strategy.open_pos(ticker=ticker, volume=abs(volume), trade_side=np.sign(volume))

    def buy(self, ticker: str, volume: float = None, limit_price: float = None):
        if ticker not in self.subscription:
            raise ValueError(f'{ticker} not subscribed for trading!')

        super().buy(ticker=ticker, volume=volume, limit_price=limit_price)

    def sell(self, ticker: str, volume: float = None, limit_price: float = None):
        if ticker not in self.subscription:
            raise ValueError(f'{ticker} not subscribed for trading!')

        super().sell(ticker=ticker, volume=volume, limit_price=limit_price)

    def run(self, **kwargs):
        if not self.event_engine.active:
            self.event_engine.start()

        replay = ProgressiveReplay(
            loader=self.load_data,
            tickers=list(self.subscription),
            dtype=['TickData', 'TradeData'],
            start_date=self.start_date,
            end_date=self.end_date,
            bod=self.bod,
            eod=self.eod,
            tick_size=kwargs.get('progress_tick_size', 0.001),
        )

        _start_ts = time.time()

        for market_data in replay:
            if self.multi_threading:
                self.lock.acquire()
                self.event_engine.put(topic=self.topic_set.push(market_data=market_data), market_data=market_data)
            else:
                self.strategy.mds.on_market_data(market_data=market_data)
                self.strategy.position_tracker.on_market_data(market_data=market_data)
                self.strategy.on_market_data(market_data=market_data)

                if market_data.ticker in self.subscription:
                    self.sim_match[market_data.ticker](market_data=market_data)
                    self.web_app.update(market_data=market_data)

            self.timestamp = market_data.timestamp
            self.last_price[market_data.ticker] = market_data.market_price

        LOGGER.info(f'All done! time_cost: {time.time() - _start_ts:,.3}s')
