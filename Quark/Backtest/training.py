__package__ = 'Quark.Backtest'

import datetime
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from AlgoEngine.Strategies import STRATEGY_ENGINE
from PyQuantKit import TransactionSide

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from . import LOGGER, simulated_env, factor_pool
from ..Base import GlobalStatics
from ..DecisionCore.Linear import *
from ..Strategy import StrategyMetric

DATA_SOURCE = pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res')
DATA_CORE = RidgeLinearCore(ticker='Synthetic', ridge_alpha=100, pred_length=30 * 60, smooth_look_back=0)
EXPORT_DIR = pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, f'{DATA_CORE.__class__.__name__}')

START_DATE = datetime.date(2023, 1, 1)
END_DATE = datetime.date(2023, 6, 1)
CALENDAR = simulated_env.trade_calendar(start_date=START_DATE, end_date=END_DATE)
FACTOR_POOL = factor_pool.FACTOR_POOL


class TradeMetrics(object):
    def __init__(self):
        self.trades = []
        self.trade_batch = []

        self.exposure = 0.
        self.cash_flow = 0.
        self.pnl = 0.

        self.current_trade_batch = {}
        self.current_price = None

    def add_trades(self, volume: float, price: float, trade_time: datetime.datetime):

        if not volume:
            return

        self.exposure += volume
        self.cash_flow -= volume * price
        self.pnl = self.exposure * price + self.cash_flow
        self.current_price = price

        self.trades.append(
            dict(
                trade_time=trade_time,
                volume=volume,
                price=price,
                exposure=self.exposure,
                cash_flow=self.cash_flow,
                pnl=self.pnl,
            )
        )

        if 'init_side' not in self.current_trade_batch:
            self.current_trade_batch['init_side'] = 1 if volume > 0 else -1

        self.current_trade_batch['cash_flow'] = self.current_trade_batch.get('cash_flow', 0.) - volume * price
        self.current_trade_batch['pnl'] = self.current_trade_batch.get('pnl', 0.) + self.exposure * price + self.current_trade_batch['cash_flow']
        self.current_trade_batch['turnover'] = self.current_trade_batch.get('turnover', 0.) + abs(volume) * price

        if not self.exposure:
            self.trade_batch.append(self.current_trade_batch)
            self.current_trade_batch = {}

    def add_trades_batch(self, trade_logs: pd.DataFrame):
        for timestamp, row in trade_logs.iterrows():  # type: float, dict
            price = row['current_price']
            volume = row['signal']
            trade_time = datetime.datetime.fromtimestamp(timestamp)

            self.add_trades(volume=volume, price=price, trade_time=trade_time)

    @property
    def info(self):

        info_dict = dict(
            total_gain=0.,
            total_loss=0.,
            trade_count=0,
            win_count=0,
            lose_count=0,
            turnover=0.,
        )

        for trade_batch in self.trade_batch:
            if trade_batch['pnl'] > 0:
                info_dict['total_gain'] += trade_batch['pnl']
                info_dict['trade_count'] += 1
                info_dict['win_count'] += 1
                info_dict['turnover'] += trade_batch['turnover']
            else:
                info_dict['total_loss'] += trade_batch['pnl']
                info_dict['trade_count'] += 1
                info_dict['lose_count'] += 1
                info_dict['turnover'] += trade_batch['turnover']

        info_dict['win_rate'] = info_dict['win_count'] / info_dict['trade_count'] if info_dict['trade_count'] else 0.
        info_dict['average_gain'] = info_dict['total_gain'] / info_dict['win_count'] / self.current_price if info_dict['win_count'] else 0.
        info_dict['average_loss'] = info_dict['total_loss'] / info_dict['lose_count'] / self.current_price if info_dict['lose_count'] else 0.
        info_dict['gain_loss_ratio'] = -info_dict['average_gain'] / info_dict['average_loss'] if info_dict['average_loss'] else 1.
        info_dict['long_avg_pnl'] = np.average([_['pnl'] for _ in long_trades]) / self.current_price if (long_trades := [_ for _ in self.trade_batch if _['init_side'] == 1]) else np.nan
        info_dict['short_avg_pnl'] = np.average([_['pnl'] for _ in short_trades]) / self.current_price if (short_trades := [_ for _ in self.trade_batch if _['init_side'] == -1]) else np.nan

        return info_dict

    def to_string(self) -> str:
        metric_info = self.info

        fmt_dict = {
            'total_gain': f'{metric_info["total_gain"]:,.3f}',
            'total_loss': f'{metric_info["total_loss"]:,.3f}',
            'trade_count': f'{metric_info["trade_count"]:,}',
            'win_count': f'{metric_info["win_count"]:,}',
            'lose_count': f'{metric_info["lose_count"]:,}',
            'turnover': f'{metric_info["turnover"]:,.3f}',
            'win_rate': f'{metric_info["win_rate"]:.2%}',
            'average_gain': f'{metric_info["average_gain"]:,.4%}',
            'average_loss': f'{metric_info["average_loss"]:,.4%}',
            'long_avg_pnl': f'{metric_info["long_avg_pnl"]:,.4%}',
            'short_avg_pnl': f'{metric_info["short_avg_pnl"]:,.4%}',
            'gain_loss_ratio': f'{metric_info["gain_loss_ratio"]:,.3%}'
        }

        return 'Trade Metrics Report:\n' + pd.Series(fmt_dict).to_string()


TRADE_METRICS = TradeMetrics()


def training(market_date: datetime.date, trace_back: int = 30):
    """
    train the data core with factor collected from {market_date} and {trace_back} days before
    """

    caches = FACTOR_POOL.locate_caches(market_date=market_date, size=trace_back, exclude_current=False)
    data_list = []

    if not caches:
        raise ValueError(f'No factor cache found at {market_date}')

    for cache_file in caches:
        info_df = DATA_CORE.load_info_from_csv(file_path=cache_file)
        data_list.append(info_df)

    report = DATA_CORE.calibrate(factor_cache=data_list, trace_back=0)

    return report


def metric_signal(market_date: datetime.date, fake_trades: bool = True):
    """
    use the data_core to generate trade signal and collect the metrics
    """

    caches = FACTOR_POOL.locate_caches(market_date=market_date, size=1, exclude_current=False)
    info_df = DATA_CORE.load_info_from_csv(file_path=caches[0])
    strategy_metric = StrategyMetric(sample_interval=1)
    position_tracker = STRATEGY_ENGINE.position_tracker

    for timestamp, row in info_df.iterrows():

        if datetime.datetime.fromtimestamp(timestamp).time() < datetime.time(9, 30):
            continue

        index_price = strategy_metric.last_assets_price = row['SyntheticIndex.Price']
        signal = DATA_CORE.signal(position=position_tracker, factor=row.to_dict(), timestamp=timestamp)
        strategy_metric.collect_signal(signal=signal, timestamp=timestamp)

        if fake_trades and signal:
            position_tracker.add_exposure(ticker=DATA_CORE.ticker, volume=1, notional=index_price, timestamp=timestamp, side=TransactionSide(signal))
            LOGGER.info(f'{datetime.datetime.fromtimestamp(timestamp)} {DATA_CORE.__class__.__name__} instructed {TransactionSide(signal).side_name} signal')

    return strategy_metric


def validate(market_date: datetime.date):
    """
    validate fitted params with factors collected from {market_date}, and plot the result
    """
    caches = FACTOR_POOL.locate_caches(market_date=market_date, size=1, exclude_current=False)
    info_df = DATA_CORE.load_info_from_csv(file_path=caches[0])
    DATA_CORE._prepare(factors=info_df)
    DATA_CORE._validate(info=info_df)
    fig = DATA_CORE.plot(info=info_df)
    return fig


def annotate_signals(fig, strategy_metric):
    metrics_df = pd.DataFrame(strategy_metric.metrics)

    # add some annotations
    for timestamp, row in metrics_df.iterrows():  # type: float, dict
        price = row['current_price']
        action = row['signal']

        if not action:
            continue

        fig.add_annotation(
            x=datetime.datetime.fromtimestamp(timestamp),  # x-coordinate
            y=price,  # y-coordinate (relative to y-axis 1)
            xref='x',
            yref='y',
            text='Sell' if action < 0 else "Buy",
            showarrow=True,
            arrowhead=3,  # red arrow shape
            # ax=20 if action < 0 else -20,  # arrow x-direction offset
            # ay=-40,  # arrow y-direction offset
            bgcolor='red' if action < 0 else 'green',
            opacity=0.8
        )

    # Export the figure to HTML
    figure_html = fig.to_html(full_html=False)

    # Export the DataFrame to HTML
    if not metrics_df.empty:
        metrics_df = metrics_df.loc[metrics_df['signal'] != 0]
        metrics_df['signal_time'] = [datetime.datetime.fromtimestamp(_) for _ in metrics_df.index]
        formatted_df = metrics_df.applymap(lambda x:
                                           f"{x:.3f}" if isinstance(x, float) else
                                           f"{x:%H:%M:%S}" if isinstance(x, datetime.datetime) else
                                           x)
        table = go.Table(
            header=dict(values=list(formatted_df.columns)),
            cells=dict(values=[formatted_df[col] for col in formatted_df.columns])
        )
        table_html = go.Figure(data=[table]).to_html(full_html=False)

        # Concatenate the HTML codes
        figure_html += '\n' + table_html

    return figure_html


def main():
    os.makedirs(EXPORT_DIR, exist_ok=True)

    for market_date in CALENDAR:
        calendar_index = CALENDAR.index(market_date)

        if calendar_index == 0:
            continue

        previous_market_date = CALENDAR[calendar_index - 1]

        cal_report = training(market_date=previous_market_date)
        strategy_metric = metric_signal(market_date=market_date)
        fig = validate(market_date=market_date)
        TRADE_METRICS.add_trades_batch(pd.DataFrame(strategy_metric.metrics))
        html_content = annotate_signals(fig=fig, strategy_metric=strategy_metric)

        with open(EXPORT_DIR.joinpath(f'{market_date}.validation.html'), 'w', encoding="utf-8") as file:
            file.write(html_content)

        LOGGER.info(f'Factor cache: {DATA_SOURCE}\ncalibration report:\n' + '\n'.join([f"{_}: {cal_report[_]}" for _ in cal_report]))
        LOGGER.info(f'\n{TRADE_METRICS.to_string()}')
        result = DATA_CORE.to_json(fmt='dict')
        result.update(TRADE_METRICS.info)

        with open(EXPORT_DIR.joinpath(f'{market_date}.{DATA_CORE.__class__.__name__}.json'), 'w', encoding="utf-8") as file:
            file.write(json.dumps(result))

        DATA_CORE.clear()


if __name__ == '__main__':
    main()
