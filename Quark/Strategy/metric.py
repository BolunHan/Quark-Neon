import pathlib

import numpy as np
import pandas as pd
from AlgoEngine.Engine import MarketDataMonitor


class StrategyMetric(object):
    def __init__(self, sample_interval: int):
        self.sample_interval = sample_interval

        self._last_update = 0.
        self.factor_value = {}
        self.assets_value = {}
        self.metrics = {
            'exposure': {},
            'cash_flow': {},
            'total_pnl': {},
            'current_pnl': {},
            'current_cash_flow': {},
            'signal_count': {},
        }

        self.signal = 0
        self.signal_accumulated = 0
        self.last_cash_flow = 0.
        self.signal_count = 0.
        self.total_cash_flow = 0.
        self.last_assets_price = 1.

    def collect_factors(self, monitors: dict[str, MarketDataMonitor], timestamp: float) -> dict[str, float]:
        factors = {}

        if monitor := monitors.get('Monitor.TradeFlow'):
            if monitor.is_ready:
                value = monitor.value

                for ticker in value:
                    factors[f'Monitor.TradeFlow.{ticker}'] = value[ticker]

        if monitor := monitors.get('Monitor.TradeFlow.EMA'):
            if monitor.is_ready:
                value = monitor.value

                for ticker in value:
                    factors[f'Monitor.TradeFlow.EMA.{ticker}'] = value[ticker]
                factors[f'Monitor.TradeFlow.EMA.Sum'] = sum(value.values())

        if monitor := monitors.get('Monitor.Coherence.Price'):
            if monitor.is_ready:
                up_dispersion, down_dispersion = monitor.value
                factors[f'Monitor.Coherence.Price.Up'] = up_dispersion
                factors[f'Monitor.Coherence.Price.Down'] = down_dispersion

        if monitor := monitors.get('Monitor.SyntheticIndex'):
            if monitor.is_ready:
                self.last_assets_price = factors[f'Monitor.SyntheticIndex.Price'] = monitor.active_bar.market_price
                factors[f'Monitor.SyntheticIndex.Notional'] = monitor.active_bar.notional
                factors[f'Monitor.SyntheticIndex.Volume'] = monitor.active_bar.volume
                factors[f'Monitor.SyntheticIndex.LastNotional'] = monitor.value.notional
                factors[f'Monitor.SyntheticIndex.LastVolume'] = monitor.value.volume

        if monitor := monitors.get('Monitor.Coherence.Volume'):
            if monitor.is_ready:
                factors[f'Monitor.Coherence.Volume'] = monitor.value

        if monitor := monitors.get('Monitor.TA.MACD'):
            if monitor.is_ready:
                value = monitor.value

                for ticker in value:
                    factors[f'Monitor.TA.MACD.{ticker}'] = value[ticker]

        if monitor := monitors.get('Monitor.TA.MACD.Index'):
            if monitor.is_ready:
                factors[f'Monitor.TA.MACD.Index'] = monitor.value

        if monitor := monitors.get('Monitor.Aggressiveness'):
            if monitor.is_ready:
                aggressive_buy, aggressive_sell = monitor.value

                for ticker in aggressive_buy:
                    factors[f'Monitor.Aggressiveness.Buy.{ticker}'] = aggressive_buy[ticker]
                    factors[f'Monitor.Aggressiveness.Sell.{ticker}'] = aggressive_sell[ticker]
                    factors[f'Monitor.Aggressiveness.Net.{ticker}'] = aggressive_buy[ticker] - aggressive_sell[ticker]
                factors[f'Monitor.Aggressiveness.Net'] = np.sum(aggressive_buy.values()) - np.sum(aggressive_sell.values())

        if monitor := monitors.get('Monitor.Aggressiveness.EMA'):
            if monitor.is_ready:
                aggressive_buy, aggressive_sell = monitor.value

                for ticker in aggressive_buy:
                    factors[f'Monitor.Aggressiveness.EMA.Buy.{ticker}'] = aggressive_buy[ticker]
                    factors[f'Monitor.Aggressiveness.EMA.Sell.{ticker}'] = aggressive_sell[ticker]
                    factors[f'Monitor.Aggressiveness.EMA.Net.{ticker}'] = aggressive_buy[ticker] - aggressive_sell[ticker]
                factors[f'Monitor.Aggressiveness.EMA.Net'] = np.sum(aggressive_buy.values()) - np.sum(aggressive_sell.values())

        if self._last_update + self.sample_interval < timestamp:
            timestamp = timestamp // self.sample_interval * self.sample_interval
            self.log_factors(factors=factors, timestamp=timestamp)
            self._last_update = timestamp

        return factors

    def log_factors(self, factors: dict, timestamp: float):
        self.assets_value[timestamp] = factors.get(f'Monitor.SyntheticIndex.Price')

        self.factor_value[timestamp] = {
            'TradeFlow.EMA.Sum': factors.get(f'Monitor.TradeFlow.EMA.Sum'),
            'Coherence.Price.Up': factors.get(f'Monitor.Coherence.Price.Up'),
            'Coherence.Price.Down': factors.get(f'Monitor.Coherence.Price.Down'),
            'Coherence.Volume': factors.get(f'Monitor.Coherence.Volume'),
            'TA.MACD.Index': factors.get(f'Monitor.TA.MACD.Index'),
            'Aggressiveness.EMA.Net': factors.get(f'Monitor.Aggressiveness.EMA.Net'),
        }

    def dump(self, file_path: str | pathlib.Path):
        info = pd.DataFrame(self.factor_value)
        info['index_value'] = pd.Series(self.assets_value)
        info.to_csv(file_path)

    def collect_signal(self, signal: int, timestamp: float):
        self.signal = signal
        self.signal_accumulated += signal

        if not self.signal_accumulated:
            self.last_cash_flow = 0.

        if signal:
            self.signal_accumulated += signal
            self.total_cash_flow -= signal * self.last_assets_price
            self.last_cash_flow -= signal * self.last_assets_price
            self.signal_count += abs(signal)

        self.metrics['exposure'][timestamp] = self.signal_accumulated
        self.metrics['cash_flow'][timestamp] = self.total_cash_flow
        self.metrics['total_pnl'][timestamp] = self.total_cash_flow + self.signal_accumulated * self.last_assets_price
        self.metrics['current_pnl'][timestamp] = self.last_cash_flow + self.signal_accumulated * self.last_assets_price
        self.metrics['current_cash_flow'][timestamp] = self.last_cash_flow
        self.metrics['signal_count'][timestamp] = self.signal_count

    def clear(self):

        self._last_update = 0.
        self.factor_value.clear()
        self.assets_value.clear()

        self.metrics['exposure'].clear()
        self.metrics['cash_flow'].clear()
        self.metrics['total_pnl'].clear()
        self.metrics['current_pnl'].clear()
        self.metrics['current_cash_flow'].clear()
        self.metrics['signal_count'].clear()

        self.signal = 0
        self.signal_accumulated = 0
        self.last_cash_flow = 0.
        self.signal_count = 0.
        self.total_cash_flow = 0.
        self.last_assets_price = 1.
