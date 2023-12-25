"""
this script is designed to validate single factor, with linear regression
"""
__package__ = 'Quark.Factor'

import datetime
import os
import pathlib
import uuid

import numpy as np
import pandas as pd
from AlgoEngine.Engine import MarketDataMonitor, ProgressiveReplay
from PyQuantKit import MarketData, BarData

from . import LOGGER, MDS, future, IndexWeight
from .LowPass import IndexMACDTriggerMonitor
from .Misc import SyntheticIndexMonitor
from ..API import historical
from ..Backtest import simulated_env
from ..Base import safe_exit, GlobalStatics
from ..Calibration.bootstrap import BootstrapLinearRegression
from ..Calibration.cross_validation import CrossValidation
from ..Misc import helper

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
        self.index_name = '000016.SH'
        self.index_weights = IndexWeight(index_name='000016.SH')
        # params for replay
        self.dtype = kwargs.get('dtype', 'TradeData')
        self.start_date = kwargs.get('start_date', datetime.date(2023, 1, 1))
        self.end_date = kwargs.get('end_date', datetime.date(2023, 2, 1))

        # params for sampling
        self.sampling_interval = kwargs.get('sampling_interval', 10.)

        # params for validation
        self.pred_target = 'Synthetic.market_price'
        self.factor_name = ['Monitor.MACD.Index.Trigger.Synthetic', 'bias']

        self.factor: MarketDataMonitor | None = None
        self.synthetic: SyntheticIndexMonitor | None = None
        self.subscription = set()
        self.metrics: dict[float, dict[str, float]] = {}

    def init_factor(self, **kwargs) -> MarketDataMonitor:
        self.factor = IndexMACDTriggerMonitor(
            weights=self.index_weights,
            update_interval=kwargs.get('update_interval', 60),
            observation_window=kwargs.get('observation_window', 5),
            confirmation_threshold=kwargs.get('confirmation_threshold', 0.)
        )

        self.synthetic = SyntheticIndexMonitor(
            index_name='Synthetic',
            weights=self.index_weights

        )

        MDS.add_monitor(self.factor)
        MDS.add_monitor(self.synthetic)

        return self.factor

    def bod(self, market_date: datetime.date, **kwargs) -> None:
        LOGGER.info(f'Starting {market_date} bod process...')

        index_weight = IndexWeight(
            index_name=self.index_name,
            **helper.load_dict(
                file_path=pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res', f'index_weights.{self.index_name}.{market_date:%Y%m%d}.json'),
                json_dict=simulated_env.query_index_weights(index_name=self.index_name, market_date=market_date)
            )
        )

        # a lite setting for fast debugging
        # for _ in list(index_weight.keys())[5:]:
        #     index_weight.pop(_)

        # step 0: update index weights
        self.index_weights.update(index_weight)
        self.index_weights.normalize()

        # startup task 1: update subscription
        subscription = set(self.index_weights.keys())

        # backtest specific action 1: unzip data
        historical.unzip_batch(market_date=market_date, ticker_list=subscription)
        replay = kwargs.get('replay')

        for _ in subscription:
            if _ not in self.subscription:
                replay.add_subscription(ticker=_, dtype='TradeData')

        for _ in self.subscription:
            if _ not in subscription:
                replay.remove_subscription(ticker=_, dtype='TradeData')

        self.subscription.update(subscription)

    def eod(self, market_date: datetime.date, **kwargs) -> None:
        LOGGER.info(f'Starting {market_date} eod process...')

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
            tickers=[],
            dtype=self.dtype.split(','),
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
        LOGGER.info(f'{market_date} validation started with {len(self.metrics):,} obs.')

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

        # Drop rows with NaN or infinite values horizontally from x, y, and x_axis
        valid_mask = np.all(np.isfinite(x), axis=1) & np.isfinite(y) & np.isfinite(x_axis)
        x = x[valid_mask]
        y = y[valid_mask]
        x_axis = x_axis[valid_mask]

        cv = CrossValidation(model=regression, folds=10, shuffle=True, strict_no_future=True)
        cv.validate(x=x, y=y)
        fig = cv.plot(x=x, y=y, x_axis=x_axis)

        # step 3: plot the results
        fig.update_xaxes(rangebreaks=[dict(bounds=[11.5, 13], pattern="hour"), dict(bounds=[15, 9.5], pattern="hour")])

        # step 4: dump data
        factor_metrics.to_csv(pathlib.Path(dump_dir, f'{market_date:%Y-%m-%d}', f'{self.factor.name}.validation.csv'))
        fig.write_html(pathlib.Path(dump_dir, f'{market_date:%Y-%m-%d}', f'{self.factor.name}.validation.html'), include_plotlyjs='cdn')

    def run(self):
        self.init_factor()
        replay = self.init_replay()

        last_update = 0.
        last_log_entry = None

        for market_data in replay:  # type: MarketData
            if not cn_market_session_filter(market_data.timestamp):
                continue

            MDS.on_market_data(market_data=market_data)
            timestamp = market_data.timestamp

            if timestamp >= last_update + self.sampling_interval:
                timestamp_index = timestamp // self.sampling_interval * self.sampling_interval

                factor_value = self.factor.value

                if factor_value is None:
                    continue

                self.metrics[timestamp_index] = last_log_entry = {}

                if isinstance(factor_value, dict):
                    for key in factor_value:
                        last_log_entry[f'{self.factor.name}.{key}'] = factor_value[key]
                elif isinstance(factor_value, BarData):
                    last_log_entry[f'{factor_value.ticker}.market_price'] = factor_value.close_price

                last_update = timestamp_index

            if last_log_entry is not None and (key := f'{market_data.ticker}.market_price') not in last_log_entry:
                last_log_entry[key] = market_data.market_price

            if last_log_entry is not None and (key := f'{self.synthetic.index_name}.market_price') not in last_log_entry:
                last_log_entry[key] = self.synthetic.index_price


def main():
    fv = FactorValidation()
    fv.run()
    safe_exit()


if __name__ == '__main__':
    main()
