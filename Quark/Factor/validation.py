"""
this script is designed to validate single factor, with linear regression
"""
__package__ = 'Quark.Factor'

import datetime
import os
import pathlib
import uuid
from types import SimpleNamespace

import pandas as pd
from AlgoEngine.Engine import MarketDataMonitor, ProgressiveReplay
from PyQuantKit import MarketData

from . import LOGGER, MDS, future
from .LowPass import MACDTriggerMonitor as Factor
from ..API import historical
from ..Backtest import simulated_env
from ..Base import safe_exit
from ..Calibration.bootstrap import BootstrapLinearRegression
from ..Calibration.cross_validation import CrossValidation

LOGGER = LOGGER.getChild('validation')
FACTOR: MarketDataMonitor | None = None
METRICS: dict[float, dict[str, float]] = {}
ENV: SimpleNamespace | None = None


def cn_market_session_filter(timestamp: float) -> bool:
    market_time = datetime.datetime.fromtimestamp(timestamp)

    # filter non-trading hours
    if market_time.time() < datetime.time(9, 30) \
            or datetime.time(11, 30) < market_time.time() < datetime.time(13, 0) \
            or datetime.time(15, 0) < market_time.time():
        return False

    return True


def simulate_env() -> SimpleNamespace:
    global ENV

    ENV = env = SimpleNamespace(
        id=f'{uuid.uuid4()}',
        # params for replay
        ticker='600036.SH',
        subscription='TradeData',
        start_date=datetime.date(2023, 1, 1),
        end_date=datetime.date(2023, 2, 1),
        # params for factor
        update_interval=60,
        observation_window=5,
        confirmation_threshold=0.,
        # params for sampling
        sampling_interval=10.
    )

    return env


def init_factor(env) -> MarketDataMonitor:
    global FACTOR
    FACTOR = factor = Factor(
        update_interval=60,
        observation_window=5,
        confirmation_threshold=0.00001
    )

    MDS.add_monitor(factor)
    return factor


def bod(market_date: datetime.date, **kwargs) -> None:
    LOGGER.info(f'Starting {market_date} bod process')


def eod(market_date: datetime.date, **kwargs) -> None:
    LOGGER.info(f'Starting {market_date} eod process')

    validation(factor=FACTOR, metrics=METRICS, market_date=market_date, dump_dir=f'Validation.{ENV.id.split("-")[0]}')

    FACTOR.clear()
    METRICS.clear()


def init_replay(env, factor: MarketDataMonitor) -> ProgressiveReplay:
    calendar = simulated_env.trade_calendar(start_date=env.start_date, end_date=env.end_date)

    replay = ProgressiveReplay(
        loader=historical.loader,
        tickers=env.ticker.split(','),
        dtype=env.subscription.split(','),
        start_date=env.start_date,
        end_date=env.end_date,
        calendar=calendar,
        bod=bod,
        eod=eod,
        tick_size=0.001,
    )

    os.makedirs(f'Validation.{env.id.split("-")[0]}', exist_ok=True)
    return replay


def validation(factor: MarketDataMonitor, metrics: dict[float, dict[str, float]], market_date: datetime.date, dump_dir: str | pathlib.Path):
    LOGGER.info(f'{market_date} {factor.name} validation started with {len(metrics):,} obs.')
    pred_target = '600036.SH.market_price'
    factor_name = ['600036.SH', 'bias']

    os.makedirs(pathlib.Path(dump_dir, f'{market_date:%Y-%m-%d}'))

    if not metrics:
        return

    # step 1: add define prediction target
    factor_metrics = pd.DataFrame(metrics).T
    factor_metrics['market_time'] = [datetime.datetime.fromtimestamp(_) for _ in factor_metrics.index]
    factor_metrics['bias'] = 1.

    future.fix_prediction_target(
        factors=factor_metrics,
        key=pred_target,
        session_filter=cn_market_session_filter,
        inplace=True,
        pred_length=15 * 60
    )

    future.wavelet_prediction_target(
        factors=factor_metrics,
        key=pred_target,
        session_filter=cn_market_session_filter,
        inplace=True,
        decode_level=4
    )

    # step 2: regression analysis
    regression = BootstrapLinearRegression()
    x_axis = factor_metrics['market_time']
    x = factor_metrics[factor_name].to_numpy()
    y = factor_metrics['up_smoothed'].to_numpy() + factor_metrics['down_smoothed'].to_numpy()
    cv = CrossValidation(model=BootstrapLinearRegression(), folds=10, shuffle=True)
    cv.validate(x=x, y=y)
    fig = cv.plot(x=x, y=y, x_axis=x_axis)

    # step 3: plot the results
    fig.update_xaxes(rangebreaks=[dict(bounds=[11.5, 13], pattern="hour"), dict(bounds=[15, 9.5], pattern="hour")])
    # fig.show()

    # step 4: dump data
    factor_metrics.to_csv(pathlib.Path(dump_dir, f'{market_date:%Y-%m-%d}', f'{factor.name}.csv'))
    fig.write_html(pathlib.Path(dump_dir, f'{market_date:%Y-%m-%d}', f'{factor.name}.html'), include_plotlyjs='cdn')


def main():
    env = simulate_env()
    factor = init_factor(env=env)
    task = init_replay(env=env, factor=factor)

    last_update = 0.
    last_log_entry = None

    for _ in task:  # type: MarketData
        if not cn_market_session_filter(_.timestamp):
            continue

        MDS.on_market_data(market_data=_)
        timestamp = _.timestamp

        if timestamp >= last_update + env.sampling_interval:
            timestamp_index = timestamp // env.sampling_interval * env.sampling_interval
            factor_value = factor.value

            METRICS[timestamp_index] = last_log_entry = {}
            last_log_entry.update(factor_value)

            last_update = timestamp_index

        if last_log_entry is not None and (key := f'{_.ticker}.market_price') not in last_log_entry:
            last_log_entry[key] = _.market_price

    safe_exit()


if __name__ == '__main__':
    main()
