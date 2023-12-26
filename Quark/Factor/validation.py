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

from . import LOGGER, MDS, future, IndexWeight, ALPHA_0001
from .Correlation import EntropyEMAMonitor
from .LowPass import IndexMACDTriggerMonitor
from .Misc import SyntheticIndexMonitor
from ..API import historical
from ..Backtest import simulated_env
from ..Base import safe_exit, GlobalStatics
from ..Calibration.bootstrap import BootstrapLinearRegression
from ..Calibration.cross_validation import CrossValidation
from ..Misc import helper

LOGGER = LOGGER.getChild('validation')
DUMMY_WEIGHT = True


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

        # params for index
        self.index_name = kwargs.get('index_name', '000016.SH')
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
        self.replay: ProgressiveReplay | None = None
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

    def _update_index_weights(self, market_date: datetime.date):
        index_weight = IndexWeight(
            index_name=self.index_name,
            **helper.load_dict(
                file_path=pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res', f'index_weights.{self.index_name}.{market_date:%Y%m%d}.json'),
                json_dict=simulated_env.query_index_weights(index_name=self.index_name, market_date=market_date)
            )
        )

        # a lite setting for fast debugging
        if DUMMY_WEIGHT:
            for _ in list(index_weight.keys())[5:]:
                index_weight.pop(_)

        # step 0: update index weights
        self.index_weights.update(index_weight)
        self.index_weights.normalize()

    def _update_subscription(self):
        subscription = set(self.index_weights.keys())

        for _ in subscription:
            if _ not in self.subscription:
                self.replay.add_subscription(ticker=_, dtype='TradeData')

        for _ in self.subscription:
            if _ not in subscription:
                self.replay.remove_subscription(ticker=_, dtype='TradeData')

        self.subscription.update(subscription)

    def bod(self, market_date: datetime.date, **kwargs) -> None:
        LOGGER.info(f'Starting {market_date} bod process...')

        # startup task 0: update subscription
        self._update_index_weights(market_date=market_date)

        # backtest specific action 1: unzip data
        historical.unzip_batch(market_date=market_date, ticker_list=self.index_weights.keys())

        # startup task 2: update replay
        self._update_subscription()

    def eod(self, market_date: datetime.date, **kwargs) -> None:
        LOGGER.info(f'Starting {market_date} eod process...')

        self.validation(market_date=market_date)

        self.reset()

    def reset(self):
        self.factor.clear()
        self.metrics.clear()

    def init_replay(self) -> ProgressiveReplay:
        calendar = simulated_env.trade_calendar(start_date=self.start_date, end_date=self.end_date)

        self.replay = ProgressiveReplay(
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

        return self.replay

    def _define_inputs(self, factors: pd.DataFrame):
        factors['market_time'] = [datetime.datetime.fromtimestamp(_) for _ in factors.index]
        factors['bias'] = 1.

    def _define_prediction(self, factors: pd.DataFrame):
        future.fix_prediction_target(
            factors=factors,
            key=self.pred_target,
            session_filter=cn_market_session_filter,
            inplace=True,
            pred_length=15 * 60
        )

        future.wavelet_prediction_target(
            factors=factors,
            key=self.pred_target,
            session_filter=cn_market_session_filter,
            inplace=True,
            decode_level=3
        )

    def _cross_validation(self, factors: pd.DataFrame):
        import plotly.graph_objects as go

        regression = BootstrapLinearRegression()
        x_axis = factors['market_time']
        x = factors[self.factor_name].to_numpy()
        y = factors['target_actual'].to_numpy()

        # Drop rows with NaN or infinite values horizontally from x, y, and x_axis
        valid_mask = np.all(np.isfinite(x), axis=1) & np.isfinite(y) & np.isfinite(x_axis)
        x = x[valid_mask]
        y = y[valid_mask]
        x_axis = x_axis[valid_mask]

        cv = CrossValidation(model=regression, folds=10, shuffle=True, strict_no_future=True)
        cv.validate(x=x, y=y)
        fig = cv.plot(x=x, y=y, x_axis=x_axis)

        candlestick_trace = go.Candlestick(
            name='Synthetic',
            x=factors['market_time'],
            open=factors['Synthetic.open_price'],
            high=factors['Synthetic.high_price'],
            low=factors['Synthetic.low_price'],
            close=factors['Synthetic.close_price'],
            yaxis='y2'
        )
        fig.add_trace(candlestick_trace)
        fig.update_xaxes(
            rangebreaks=[dict(bounds=[11.5, 13], pattern="hour"), dict(bounds=[15, 9.5], pattern="hour")],
        )
        fig.update_layout(
            yaxis2=dict(
                title="Synthetic",
                overlaying='y',
                side='right',
                showgrid=False
            )
        )

        return cv, fig

    def _dump_result(self, market_date: datetime.date, factors: pd.DataFrame, fig):
        dump_dir = f'Validation.{self.validation_id.split("-")[0]}'
        os.makedirs(dump_dir, exist_ok=True)

        entry_dir = pathlib.Path(dump_dir, f'{market_date:%Y-%m-%d}')
        os.makedirs(entry_dir, exist_ok=True)

        factors.to_csv(pathlib.Path(entry_dir, f'{self.factor.name}.validation.csv'))
        fig.write_html(pathlib.Path(entry_dir, f'{self.factor.name}.validation.html'), include_plotlyjs='cdn')

    def validation(self, market_date: datetime.date):
        if not self.metrics:
            return

        LOGGER.info(f'{market_date} validation started with {len(self.metrics):,} obs.')

        # step 1: add define prediction target
        factor_metrics = pd.DataFrame(self.metrics).T

        self._define_inputs(factors=factor_metrics)
        self._define_prediction(factors=factor_metrics)

        # step 2: regression analysis
        cv, fig = self._cross_validation(factors=factor_metrics)

        # step 3: dump the results
        self._dump_result(market_date=market_date, factors=factor_metrics, fig=fig)

    def _collect_synthetic(self, timestamp: float, current_bar: BarData | None, last_update: float, entry_log: dict[str, float]):
        synthetic_price = self.synthetic.index_price

        if current_bar is not None:
            current_bar.close_price = synthetic_price
            current_bar.high_price = max(current_bar.high_price, synthetic_price)
            current_bar.low_price = min(current_bar.low_price, synthetic_price)

        if timestamp >= last_update + self.sampling_interval:
            timestamp_index = timestamp // self.sampling_interval * self.sampling_interval

            if current_bar is not None:
                entry_log['Synthetic.open_price'] = current_bar.open_price
                entry_log['Synthetic.close_price'] = current_bar.close_price
                entry_log['Synthetic.high_price'] = current_bar.high_price
                entry_log['Synthetic.low_price'] = current_bar.low_price
                entry_log['Synthetic.notional'] = current_bar.notional

            current_bar = BarData(
                ticker='Synthetic',
                bar_start_time=datetime.datetime.fromtimestamp(timestamp_index),
                bar_span=datetime.timedelta(seconds=self.sampling_interval),
                open_price=synthetic_price,
                close_price=synthetic_price,
                high_price=synthetic_price,
                low_price=synthetic_price
            )

        return current_bar

    def _collect_factor(self, timestamp: float, last_update: float, entry_log: dict[str, float]):
        if timestamp >= last_update + self.sampling_interval:

            factor_value = self.factor.value

            if isinstance(factor_value, dict):
                for key in factor_value:
                    entry_log[f'{self.factor.name}.{key}'] = factor_value[key]
            elif isinstance(factor_value, BarData):
                entry_log[f'{factor_value.ticker}.market_price'] = factor_value.close_price

    def _collect_market_price(self, ticker: str, market_price: float, entry_log: dict[str, float]):
        synthetic_price = self.synthetic.index_price

        if entry_log is not None and (key := f'{ticker}.market_price') not in entry_log:
            entry_log[key] = market_price

        if entry_log is not None and (key := f'{self.synthetic.index_name}.market_price') not in entry_log:
            entry_log[key] = synthetic_price

    def run(self):
        self.init_factor()
        self.init_replay()

        last_update = 0.
        entry_log = None
        current_bar: BarData | None = None

        for market_data in self.replay:  # type: MarketData
            if not cn_market_session_filter(market_data.timestamp):
                continue

            MDS.on_market_data(market_data=market_data)

            timestamp = market_data.timestamp
            ticker = market_data.ticker
            market_price = market_data.market_price

            if timestamp >= last_update + self.sampling_interval:
                timestamp_index = timestamp // self.sampling_interval * self.sampling_interval
                self.metrics[timestamp_index] = entry_log = {}

            current_bar = self._collect_synthetic(timestamp=timestamp, current_bar=current_bar, last_update=last_update, entry_log=entry_log)
            self._collect_factor(timestamp=timestamp, last_update=last_update, entry_log=entry_log)
            self._collect_market_price(ticker=ticker, market_price=market_price, entry_log=entry_log)

            last_update = timestamp // self.sampling_interval * self.sampling_interval


class FactorBatchValidation(FactorValidation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.factor_name: list[str] = ['Monitor.MACD.Index.Trigger.Synthetic', 'Monitor.Entropy.Price.EMA', 'bias']
        self.factor: list[MarketDataMonitor] = []

    def init_factor(self, **kwargs) -> list[MarketDataMonitor]:
        self.factor = [
            IndexMACDTriggerMonitor(
                weights=self.index_weights,
                update_interval=kwargs.get('update_interval', 60),
                observation_window=kwargs.get('observation_window', 5),
                confirmation_threshold=kwargs.get('confirmation_threshold', 0.)
            ),
            EntropyEMAMonitor(
                weights=self.index_weights,
                update_interval=kwargs.get('update_interval', 60),
                alpha=ALPHA_0001,
                discount_interval=1
            )
        ]

        self.synthetic = SyntheticIndexMonitor(
            index_name='Synthetic',
            weights=self.index_weights

        )

        for _ in self.factor:
            MDS.add_monitor(_)

        MDS.add_monitor(self.synthetic)

        return self.factor

    def _collect_factor(self, timestamp: float, last_update: float, entry_log: dict[str, float]):
        if timestamp >= last_update + self.sampling_interval:

            for factor in self.factor:
                factor_value = factor.value

                if isinstance(factor_value, dict):
                    for key in factor_value:
                        entry_log[f'{factor.name}.{key}'] = factor_value[key]
                elif isinstance(factor_value, BarData):
                    entry_log[f'{factor_value.ticker}.market_price'] = factor_value.close_price
                elif isinstance(factor_value, (int, float)):
                    entry_log[f'{factor.name}'] = factor_value
                else:
                    raise NotImplementedError(f'Invalid factor value type: {type(factor_value)}')

    def _dump_result(self, market_date: datetime.date, factors: pd.DataFrame, fig):
        dump_dir = f'BatchValidation.{self.validation_id.split("-")[0]}'
        os.makedirs(dump_dir, exist_ok=True)

        entry_dir = pathlib.Path(dump_dir, f'{market_date:%Y-%m-%d}')
        os.makedirs(entry_dir, exist_ok=True)

        file_name = f'{"".join([f"[{factor.name}]" for factor in self.factor])}.validation'
        factors.to_csv(pathlib.Path(entry_dir, f'{file_name}.csv'))
        fig.write_html(pathlib.Path(entry_dir, f'{file_name}.html'), include_plotlyjs='cdn')

    def reset(self):
        for _ in self.factor:
            _.clear()
        self.synthetic.clear()


def main():
    # fv = FactorValidation()
    fv = FactorBatchValidation()
    fv.run()
    safe_exit()


if __name__ == '__main__':
    main()
