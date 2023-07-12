import datetime
import pathlib

import numpy as np
import pandas as pd
from AlgoEngine.Engine import MarketDataMonitor

from .decoder import Wavelet
from ..Base import GlobalStatics

TIME_ZONE = GlobalStatics.TIME_ZONE


class StrategyMetric(object):
    def __init__(self, sample_interval: int, index_weights: dict[str, float]):
        self.sample_interval = sample_interval
        self.index_weights = index_weights

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
        """
        collect all the factors, from the registered MarketDataMonitor
        """

        factors = {}

        if monitor := monitors.get('Monitor.TradeFlow'):
            if monitor.is_ready:
                value = monitor.value

                for ticker in value:
                    factors[f'TradeFlow.{ticker}'] = value[ticker]

        if monitor := monitors.get('Monitor.TradeFlow.EMA'):
            if monitor.is_ready:
                value = monitor.value
                weighted_sum = 0.

                for ticker in value:
                    factors[f'TradeFlow.EMA.{ticker}'] = value[ticker]

                    if ticker in self.index_weights:
                        weighted_sum += value[ticker] * self.index_weights[ticker]

                factors[f'TradeFlow.EMA.Sum'] = weighted_sum

        if monitor := monitors.get('Monitor.Coherence.Price'):
            if monitor.is_ready:
                up_dispersion, down_dispersion = monitor.value
                factors[f'Coherence.Price.Up'] = up_dispersion
                factors[f'Coherence.Price.Down'] = down_dispersion
                if up_dispersion < 0:
                    factors[f'Coherence.Price.Ratio'] = 1.
                elif down_dispersion < 0:
                    factors[f'Coherence.Price.Ratio'] = 0.
                else:
                    factors[f'Coherence.Price.Ratio'] = down_dispersion / (up_dispersion + down_dispersion)

        if monitor := monitors.get('Monitor.Coherence.Price.EMA'):
            if monitor.is_ready:
                up_dispersion, down_dispersion, dispersion_ratio = monitor.value
                factors[f'Coherence.Price.Up'] = up_dispersion
                factors[f'Coherence.Price.Down'] = down_dispersion
                factors[f'Coherence.Price.Ratio.EMA'] = dispersion_ratio

        if monitor := monitors.get('Monitor.SyntheticIndex'):
            self.last_assets_price = factors[f'SyntheticIndex.Price'] = monitor.index_price
            if monitor.is_ready:
                factors[f'SyntheticIndex.Notional'] = monitor.active_bar.notional
                factors[f'SyntheticIndex.Volume'] = monitor.active_bar.volume
                factors[f'SyntheticIndex.LastNotional'] = monitor.value.notional
                factors[f'SyntheticIndex.LastVolume'] = monitor.value.volume

        if monitor := monitors.get('Monitor.Coherence.Volume'):
            if monitor.is_ready:
                factors[f'Coherence.Volume'] = monitor.value

        if monitor := monitors.get('Monitor.TA.MACD'):
            if monitor.is_ready:
                value = monitor.value

                for ticker in value:
                    factors[f'TA.MACD.{ticker}'] = value[ticker]

                factors[f'TA.MACD.Index'] = monitor.weighted_index

        if monitor := monitors.get('Monitor.Aggressiveness'):
            if monitor.is_ready:
                aggressive_buy, aggressive_sell = monitor.value

                for ticker in aggressive_buy:
                    factors[f'Aggressiveness.Buy.{ticker}'] = aggressive_buy[ticker]

                for ticker in aggressive_sell:
                    factors[f'Aggressiveness.Sell.{ticker}'] = aggressive_sell[ticker]
                factors[f'Aggressiveness.Net'] = np.sum(list(aggressive_buy.values())) - np.sum(list(aggressive_sell.values()))

        if monitor := monitors.get('Monitor.Aggressiveness.EMA'):
            if monitor.is_ready:
                aggressive_buy, aggressive_sell = monitor.value
                weighted_sum = 0.

                for ticker in aggressive_buy:
                    factors[f'Aggressiveness.EMA.Buy.{ticker}'] = aggressive_buy[ticker]

                for ticker in aggressive_sell:
                    factors[f'Aggressiveness.EMA.Sell.{ticker}'] = aggressive_sell[ticker]

                for ticker in self.index_weights:
                    weighted_sum += (aggressive_buy.get(ticker, 0.) - aggressive_sell.get(ticker, 0.)) * self.index_weights[ticker]

                factors[f'Aggressiveness.EMA.Net'] = weighted_sum

        if monitor := monitors.get('Monitor.Entropy.Price'):
            if monitor.is_ready:
                entropy = monitor.value

                factors[f'Entropy.Price'] = entropy

        if monitor := monitors.get('Monitor.Entropy.Price.EMA'):
            if monitor.is_ready:
                entropy = monitor.value

                factors[f'Entropy.Price.EMA'] = entropy

        if monitor := monitors.get('Monitor.Volatility.Daily'):
            if monitor.is_ready:
                volatility_adjusted_range = monitor.value

                for ticker in volatility_adjusted_range:
                    factors[f'Volatility.Daily.{ticker}'] = volatility_adjusted_range.get(ticker, 0.)

                factors[f'Volatility.Daily.Index'] = monitor.weighted_index

        if monitor := monitors.get('Monitor.FactorPool.Dummy'):
            if monitor.is_ready:
                factor_cache = monitor.value
                factors.update(factor_cache)

        # update observation timestamp
        if self._last_update + self.sample_interval < timestamp:
            timestamp = timestamp // self.sample_interval * self.sample_interval
            self.select_factors(factors=factors, timestamp=timestamp)
            self._last_update = timestamp

        return factors

    def log_wavelet(self, wavelet: Wavelet):
        for ts in self.factor_value:
            if wavelet.start_ts <= ts <= wavelet.end_ts:
                self.factor_value[ts]['Decoder.Flag'] = wavelet.flag.value

    def select_factors(self, factors: dict, timestamp: float):
        """
        select of interest factors from all the factors
        """
        self.assets_value[timestamp] = factors.get(f'SyntheticIndex.Price')

        self.factor_value[timestamp] = {
            'TradeFlow.EMA.Sum': factors.get(f'TradeFlow.EMA.Sum'),
            'Coherence.Price.Up': factors.get(f'Coherence.Price.Up'),
            'Coherence.Price.Down': factors.get(f'Coherence.Price.Down'),
            'Coherence.Price.Ratio.EMA': factors.get(f'Coherence.Price.Ratio.EMA'),
            'Coherence.Volume': factors.get(f'Coherence.Volume'),
            'TA.MACD.Index': factors.get(f'TA.MACD.Index'),
            # 'Aggressiveness.Net': factors.get(f'Aggressiveness.Net'),
            'Aggressiveness.EMA.Net': factors.get(f'Aggressiveness.EMA.Net'),
            # 'Entropy.Price': factors.get(f'Entropy.Price'),
            'Entropy.Price.EMA': factors.get(f'Entropy.Price.EMA'),
            # 'Volatility.Daily.Index': factors.get(f'Volatility.Daily.Index'),
        }

        return factors

    def dump(self, file_path: str | pathlib.Path):
        info = self.info
        info.index = [datetime.datetime.fromtimestamp(_, tz=TIME_ZONE) for _ in info.index]

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

    @property
    def info(self) -> pd.DataFrame:
        info = pd.DataFrame(self.factor_value).T
        info['index_value'] = pd.Series(self.assets_value)
        return info
