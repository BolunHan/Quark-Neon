__package__ = 'Quark.Backtest'

import datetime
import os
import pathlib
import sys

import pandas as pd
import plotly.graph_objects as go
from AlgoEngine.Strategies import STRATEGY_ENGINE
from PyQuantKit import TransactionSide

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from . import LOGGER, simulated_env, factor_pool
from ..Base import GlobalStatics
from ..Calibration.linear import *
from ..Strategy import StrategyMetric

DATA_SOURCE = pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res')
DATA_CORE = RidgeCore(ticker='Synthetic')

START_DATE = datetime.date(2023, 1, 1)
END_DATE = datetime.date(2023, 6, 1)
CALENDAR = simulated_env.trade_calendar(start_date=START_DATE, end_date=END_DATE)
FACTOR_POOL = factor_pool.FACTOR_POOL


def training(market_date: datetime.date, trace_back: int = 5):
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

    LOGGER.info(f'{DATA_SOURCE} calibration report:\n' + '\n'.join([f"{_}: {report[_]}" for _ in report]))


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
    fig = DATA_CORE.plot(info=info_df, decoder=DATA_CORE.decoder)
    return fig


def annotate_signals(fig, strategy_metric):
    metrics_df = pd.DataFrame(strategy_metric.metrics)

    # add some annotations
    for timestamp, row in metrics_df.iterrows():
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
        table = go.Table(
            header=dict(values=list(metrics_df.columns)),
            cells=dict(values=[metrics_df[col] for col in metrics_df.columns])
        )
        table_html = go.Figure(data=[table]).to_html(full_html=False)

        # Concatenate the HTML codes
        figure_html += '\n' + table_html

    return figure_html


def main():
    for market_date in CALENDAR:
        calendar_index = CALENDAR.index(market_date)

        if calendar_index == 0:
            continue

        previous_market_date = CALENDAR[calendar_index - 1]

        training(market_date=previous_market_date)
        strategy_metric = metric_signal(market_date=market_date)
        fig = validate(market_date=market_date)
        html_content = annotate_signals(fig=fig, strategy_metric=strategy_metric)

        with open(os.path.realpath(f'{market_date}.validation.html'), 'w', encoding="utf-8") as file:
            file.write(html_content)


if __name__ == '__main__':
    main()
