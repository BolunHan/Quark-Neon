import os
import datetime
from collections import deque
from typing import Literal

import psutil
from algo_engine.base import MarketData, TradeData, TransactionData, TickData
from algo_engine.profile import PROFILE

from .. import LOGGER
from ..utils import FactorMonitor

__all__ = ['TelemetryMonitor']


class TelemetryMonitor(FactorMonitor):
    def __init__(self, collect_out_session: bool = True, enable_trade_monitor: bool = True, enable_tick_monitor=True, name: str = 'Monitor.Telemetry'):
        super().__init__(name=name)

        self.collect_out_session = collect_out_session
        self.enable_trade_monitor = enable_trade_monitor
        self.enable_tick_monitor = enable_tick_monitor
        self.enable_sys_monitor = enable_tick_monitor

        self._datafeed_break_threshold = 6

        self.ts = 0.
        self.market_data_statistics: dict[Literal['TradeData', 'TransactionData', 'TickData'] | str, dict[str, float, int] | dict] = {}
        self.pid = os.getpid()

    def __call__(self, market_data: MarketData, **kwargs):
        self.ts = market_data.timestamp
        super().__call__(market_data, **kwargs)

    def on_trade_data(self, trade_data: TradeData | TransactionData, **kwargs) -> None:
        if not self.enable_trade_monitor:
            return

        dtype = trade_data.__class__.__name__
        ticker = trade_data.ticker

        if dtype in self.market_data_statistics:
            trade_statistics = self.market_data_statistics[dtype]
        else:
            trade_statistics = self.market_data_statistics[dtype] = {}

        if ticker in trade_statistics:
            ticker_statistics = trade_statistics[ticker]
        else:
            ticker_statistics = trade_statistics[ticker] = {
                'log_id': set(),
                'count': 0,
                'acc_volume': 0
            }

        trade_id_set: set[str | int | tuple[str | int, str | int]] = ticker_statistics['log_id']

        # trade_id = trade_data.transaction_id
        trade_id = (trade_data.buy_id, trade_data.sell_id)
        if trade_id in trade_id_set:
            LOGGER.error(f'Duplicated {dtype} {trade_data} with id transaction_id={trade_data.transaction_id}, buy_id={trade_data.buy_id}, sell_id={trade_data.sell_id}!')
            return

        trade_id_set.add(trade_id)
        ticker_statistics['count'] += 1
        ticker_statistics['acc_volume'] += trade_data.volume

        if self.enable_tick_monitor:
            acc_volume_snapshot = self.market_data_statistics.get('TickData', {}).get(ticker, {}).get('acc_volume', 0)

            if ticker_statistics['acc_volume'] > acc_volume_snapshot * 1.005:
                LOGGER.error(f'{ticker} accumulated volume mismatched at {trade_data}!')

    def on_tick_data(self, tick_data: TickData, **kwargs) -> None:
        if not self.enable_trade_monitor:
            return

        dtype = tick_data.__class__.__name__
        ticker = tick_data.ticker
        timestamp = tick_data.timestamp

        if dtype in self.market_data_statistics:
            tick_statistics = self.market_data_statistics[dtype]
        else:
            tick_statistics = self.market_data_statistics[dtype] = {}

        if ticker in tick_statistics:
            ticker_statistics = tick_statistics[ticker]
        else:
            ticker_statistics = tick_statistics[ticker] = {
                'timestamp': deque(),
                'count': 0,
                'acc_volume': 0
            }

        last_ts = ticker_statistics['timestamp'][-1] if ticker_statistics['count'] else 0

        if last_ts and timestamp - last_ts > self._datafeed_break_threshold:
            LOGGER.error(f'{ticker} breaking tick data feed detected at {tick_data.market_time}, last_update={datetime.datetime.fromtimestamp(last_ts, tz=PROFILE.time_zone)}, break seconds = {timestamp - last_ts:,.2f}!')
        elif last_ts >= timestamp:
            LOGGER.error(f'{ticker} duplicated tick data feed detected at {timestamp}!')
            return

        ticker_statistics['timestamp'].append(timestamp)
        ticker_statistics['count'] += 1
        ticker_statistics['acc_volume'] += tick_data.total_traded_volume

    def factor_names(self, subscription: list[str]) -> list[str]:
        return []

    @property
    def value(self) -> dict[str, float] | float:
        info: dict[str, int | float] = {}

        if self.enable_sys_monitor:
            process = psutil.Process(self.pid)
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent(interval=0)
            info.update(
                memory=memory_info.rss,
                cpu_percent=cpu_percent
            )

        if 'TradeData' in self.market_data_statistics:
            info.update(
                ttl_trade_count=sum([statistics.get('count') for ticker, statistics in self.market_data_statistics.get('TradeData').items()]),
                ttl_trade_volume=sum([statistics.get('acc_volume') for ticker, statistics in self.market_data_statistics.get('TradeData').items()])
            )
        elif 'TransactionData' in self.market_data_statistics:
            info.update(
                ttl_trade_count=sum([statistics.get('count') for ticker, statistics in self.market_data_statistics.get('TransactionData').items()]),
                ttl_trade_volume=sum([statistics.get('acc_volume') for ticker, statistics in self.market_data_statistics.get('TransactionData').items()])
            )

        if 'TickData' in self.market_data_statistics:
            info.update(
                ttl_tick_count=sum([statistics.get('count') for ticker, statistics in self.market_data_statistics.get('TickData').items()]),
                ttl_tick_volume=sum([statistics.get('acc_volume') for ticker, statistics in self.market_data_statistics.get('TickData').items()])
            )

        return info
