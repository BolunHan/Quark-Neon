"""
this script is designed to validate single factor, with linear regression
"""
__package__ = 'Quark.Factor'

import datetime
import os
import pathlib
import uuid

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


def cn_market_session_filter(timestamp: float) -> bool:
    market_time = datetime.datetime.fromtimestamp(timestamp)

    # filter non-trading hours
    if market_time.time() < datetime.time(9, 30) \
            or datetime.time(11, 30) < market_time.time() < datetime.time(13, 0) \
            or datetime.time(15, 0) < market_time.time():
        return False

    return True


class FactorValidation(object):
    """
    Class for performing factor validation with replay and regression analysis.

    Attributes:
        validation_id (str): Identifier for the validation instance.
        ticker (str): Ticker for market data replay.
        subscription (str): Market data subscription type.
        start_date (datetime.date): Start date for the replay.
        end_date (datetime.date): End date for the replay.
        sampling_interval (float): Interval for sampling market data.
        pred_target (str): Prediction target for validation.
        factor_name (list): Names of factors for validation.
        factor (MarketDataMonitor): Market data monitor for factor validation.
        metrics (dict): Dictionary to store validation metrics.

    Methods:
        __init__(self, **kwargs): Initialize the FactorValidation instance.
        init_factor(self, **kwargs): Initialize the factor for validation.
        bod(self, market_date: datetime.date, **kwargs) -> None: Execute beginning-of-day process.
        eod(self, market_date: datetime.date, **kwargs) -> None: Execute end-of-day process.
        init_replay(self) -> ProgressiveReplay: Initialize market data replay.
        validation(self, market_date: datetime.date, dump_dir: str | pathlib.Path): Perform factor validation.
        run(self): Run the factor validation process.

    Note: Cross validation is facing future function issue.
    """

    def __init__(self, **kwargs):

        self.validation_id = kwargs.get('validation_id', f'{uuid.uuid4()}')

        # params for replay
        self.ticker = kwargs.get('ticker', '600036.SH')
        self.subscription = kwargs.get('subscription', 'TradeData')
        self.start_date = kwargs.get('start_date', datetime.date(2023, 1, 1))
        self.end_date = kwargs.get('end_date', datetime.date(2023, 2, 1))

        # params for sampling
        self.sampling_interval = kwargs.get('sampling_interval', 10.)

        # params for validation
        self.pred_target = '600036.SH.market_price'
        self.factor_name = ['Monitor.MACD.Trigger.600036.SH', 'bias']

        self.factor: MarketDataMonitor | None = self.init_factor(**kwargs)
        self.metrics: dict[float, dict[str, float]] = {}

    def init_factor(self, **kwargs):
        factor = Factor(
            update_interval=kwargs.get('update_interval', 60),
            observation_window=kwargs.get('observation_window', 5),
            confirmation_threshold=kwargs.get('confirmation_threshold', 0.)
        )

        MDS.add_monitor(factor)
        return factor

    def bod(self, market_date: datetime.date, **kwargs) -> None:
        LOGGER.info(f'Starting {self.factor.name} {market_date} bod process...')

    def eod(self, market_date: datetime.date, **kwargs) -> None:
        LOGGER.info(f'Starting {self.factor.name} {market_date} eod process...')

        self.validation(
            market_date=market_date,
            dump_dir=f'Validation.{self.validation_id.split("-")[0]}'
        )

        self.factor.clear()
        self.metrics.clear()

    def init_replay(self) -> ProgressiveReplay:
        calendar = simulated_env.trade_calendar(start_date=self.start_date, end_date=self.end_date)

        replay = ProgressiveReplay(
            loader=historical.loader,
            tickers=self.ticker.split(','),
            dtype=self.subscription.split(','),
            start_date=self.start_date,
            end_date=self.end_date,
            calendar=calendar,
            bod=self.bod,
            eod=self.eod,
            tick_size=0.001,
        )

        os.makedirs(f'Validation.{self.validation_id.split("-")[0]}', exist_ok=True)
        return replay

    def validation(self, market_date: datetime.date, dump_dir: str | pathlib.Path):
        LOGGER.info(f'{market_date} {self.factor.name} validation started with {len(self.metrics):,} obs.')

        os.makedirs(pathlib.Path(dump_dir, f'{market_date:%Y-%m-%d}'))

        if not self.metrics:
            return

        # step 1: add define prediction target
        factor_metrics = pd.DataFrame(self.metrics).T
        factor_metrics['market_time'] = [datetime.datetime.fromtimestamp(_) for _ in factor_metrics.index]
        factor_metrics['bias'] = 1.

        future.fix_prediction_target(
            factors=factor_metrics,
            key=self.pred_target,
            session_filter=cn_market_session_filter,
            inplace=True,
            pred_length=15 * 60
        )

        future.wavelet_prediction_target(
            factors=factor_metrics,
            key=self.pred_target,
            session_filter=cn_market_session_filter,
            inplace=True,
            decode_level=4
        )

        # step 2: regression analysis
        regression = BootstrapLinearRegression()
        x_axis = factor_metrics['market_time']
        x = factor_metrics[self.factor_name].to_numpy()
        y = factor_metrics['up_smoothed'].to_numpy() + factor_metrics['down_smoothed'].to_numpy()
        cv = CrossValidation(model=regression, folds=10, shuffle=True, strict_no_future=True)
        cv.validate(x=x, y=y)
        fig = cv.plot(x=x, y=y, x_axis=x_axis)

        # step 3: plot the results
        fig.update_xaxes(rangebreaks=[dict(bounds=[11.5, 13], pattern="hour"), dict(bounds=[15, 9.5], pattern="hour")])

        # step 4: dump data
        factor_metrics.to_csv(pathlib.Path(dump_dir, f'{market_date:%Y-%m-%d}', f'{self.factor.name}.csv'))
        fig.write_html(pathlib.Path(dump_dir, f'{market_date:%Y-%m-%d}', f'{self.factor.name}.html'), include_plotlyjs='cdn')

    def run(self):
        factor = self.init_factor()
        task = self.init_replay()

        last_update = 0.
        last_log_entry = None

        for market_data in task:  # type: MarketData
            if not cn_market_session_filter(market_data.timestamp):
                continue

            MDS.on_market_data(market_data=market_data)
            timestamp = market_data.timestamp

            if timestamp >= last_update + self.sampling_interval:
                timestamp_index = timestamp // self.sampling_interval * self.sampling_interval
                factor_value = self.factor.value

                self.metrics[timestamp_index] = last_log_entry = {}

                for _ in factor_value:
                    last_log_entry[f'{factor.name}.{_}'] = factor_value[_]

                last_update = timestamp_index

            if last_log_entry is not None and (key := f'{market_data.ticker}.market_price') not in last_log_entry:
                last_log_entry[key] = market_data.market_price


def main():
    fv = FactorValidation()
    fv.run()
    safe_exit()


if __name__ == '__main__':
    main()
