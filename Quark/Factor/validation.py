"""
This script is designed for factor validation using linear regression.
"""
__package__ = 'Quark.Factor'

import datetime
import os
import pathlib
import uuid
from collections import deque

import numpy as np
import pandas as pd
from AlgoEngine.Engine import MarketDataMonitor, ProgressiveReplay
from PyQuantKit import MarketData, BarData

from . import LOGGER, MDS, future, IndexWeight, ALPHA_0001, collect_factor
from .Correlation import EntropyEMAMonitor
from .LowPass import IndexMACDTriggerMonitor
from .Misc import SyntheticIndexMonitor
from ..API import historical
from ..Backtest import simulated_env, factor_pool
from ..Base import safe_exit, GlobalStatics
from ..Calibration.Kernel import poly_kernel
from ..Calibration.bootstrap import *
from ..Calibration.cross_validation import CrossValidation
from ..Calibration.dummies import is_cn_market_session, session_dummies
from ..Misc import helper
from .decoder import RecursiveDecoder

LOGGER = LOGGER.getChild('validation')
DUMMY_WEIGHT = False
TIME_ZONE = GlobalStatics.TIME_ZONE


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
        features (list): Names of features for validation.
        factor (MarketDataMonitor): Market data monitor for factor validation.
        factor_value (dict): Dictionary to store validation metrics.

    Methods:
        __init__(self, **kwargs): Initialize the FactorValidation instance.
        init_factor(self, **kwargs): Initialize the factor for validation.
        bod(self, market_date: datetime.date, **kwargs) -> None: Execute beginning-of-day process.
        eod(self, market_date: datetime.date, **kwargs) -> None: Execute end-of-day process.
        init_replay(self) -> ProgressiveReplay: Initialize market data replay.
        validation(self, market_date: datetime.date, dump_dir: str | pathlib.Path): Perform factor validation.
        run(self): Run the factor validation process.
    """

    def __init__(self, **kwargs):
        """
        Initializes the FactorValidation instance.

        Args:
            **kwargs: Additional parameters for configuration.
        """
        self.validation_id = kwargs.get('validation_id', f'{uuid.uuid4()}')

        # Params for index
        self.index_name = kwargs.get('index_name', '000016.SH')
        self.index_weights = IndexWeight(index_name='000016.SH')

        # Params for replay
        self.dtype = kwargs.get('dtype', 'TradeData')
        self.start_date = kwargs.get('start_date', datetime.date(2023, 1, 1))
        self.end_date = kwargs.get('end_date', datetime.date(2023, 2, 1))

        # Params for sampling
        self.sampling_interval = kwargs.get('sampling_interval', 10.)

        # Params for validation
        self.decoder = RecursiveDecoder(level=3)
        self.pred_target = 'Synthetic.market_price'
        self.features = {'MACD.Index.Trigger.Synthetic'}

        self.factor: MarketDataMonitor | None = None
        self.synthetic: SyntheticIndexMonitor | None = None
        self.subscription = set()
        self.replay: ProgressiveReplay | None = None
        self.factor_value: dict[float, dict[str, float]] = {}

        self.model = RidgeRegression(alpha=0.0)
        self.coefficients: dict[str, float] = {}
        self.cv = CrossValidation(model=self.model, folds=10, shuffle=True, strict_no_future=True)

    def init_factor(self, **kwargs) -> MarketDataMonitor:
        """
        Initializes the factor for validation.

        Args:
            **kwargs: Additional parameters for factor configuration.

        Returns:
            MarketDataMonitor: Initialized market data monitor.
        """
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
        """
        Updates index weights based on the provided market date.

        Args:
            market_date (datetime.date): Date for which to update index weights.
        """
        index_weight = IndexWeight(
            index_name=self.index_name,
            **helper.load_dict(
                file_path=pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res', f'index_weights.{self.index_name}.{market_date:%Y%m%d}.json'),
                json_dict=simulated_env.query_index_weights(index_name=self.index_name, market_date=market_date)
            )
        )

        # A lite setting for fast debugging
        if DUMMY_WEIGHT:
            for _ in list(index_weight.keys())[5:]:
                index_weight.pop(_)

        # Step 0: Update index weights
        self.index_weights.update(index_weight)
        self.index_weights.normalize()

    def _update_subscription(self):
        """
        Updates market data subscriptions based on index weights.
        """
        subscription = set(self.index_weights.keys())

        for _ in subscription:
            if _ not in self.subscription:
                self.replay.add_subscription(ticker=_, dtype='TradeData')

        for _ in self.subscription:
            if _ not in subscription:
                self.replay.remove_subscription(ticker=_, dtype='TradeData')

        self.subscription.update(subscription)

    def bod(self, market_date: datetime.date, **kwargs) -> None:
        """
        Executes the beginning-of-day process.

        Args:
            market_date (datetime.date): Current market date.
            **kwargs: Additional parameters.
        """
        LOGGER.info(f'Starting {market_date} bod process...')

        # Startup task 0: Update subscription
        self._update_index_weights(market_date=market_date)

        # Backtest specific action 1: Unzip data
        historical.unzip_batch(market_date=market_date, ticker_list=self.index_weights.keys())

        # Startup task 2: Update replay
        self._update_subscription()

    def eod(self, market_date: datetime.date, **kwargs) -> None:
        """
        Executes the end-of-day process.

        Args:
            market_date (datetime.date): Current market date.
            **kwargs: Additional parameters.
        """
        LOGGER.info(f'Starting {market_date} eod process...')

        self.validation(market_date=market_date)

        self.reset()

    def reset(self):
        """
        Resets the factor and factor_value data.
        """
        self.factor.clear()
        self.factor_value.clear()
        self.decoder.clear()

    def init_replay(self) -> ProgressiveReplay:
        """
        Initializes market data replay.

        Returns:
            ProgressiveReplay: Initialized market data replay.
        """
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
        """
        Defines input features for regression analysis.

        Args:
            factors (pd.DataFrame): DataFrame containing factors.
        """
        factors['market_time'] = [datetime.datetime.fromtimestamp(_) for _ in factors.index]
        factors['bias'] = 1.

        x_matrix = factors.loc[:, list(self.features)]

        invalid_factor: pd.DataFrame = x_matrix.loc[:, x_matrix.nunique() == 1]
        if not invalid_factor.empty:
            LOGGER.error(f'Invalid factor {invalid_factor}, add epsilon to avoid multicolinearity')
            for name in invalid_factor.columns:
                x_matrix[name] = 1 + np.random.normal(scale=0.1, size=len(x_matrix))

        x_matrix['bias'] = 1.

        for _ in x_matrix.columns:
            self.coefficients[_] = 0.

        x = x_matrix.to_numpy()

        return x

    def _define_prediction(self, factors: pd.DataFrame):
        """
        Defines the prediction target for regression analysis.

        Args:
            factors (pd.DataFrame): DataFrame containing factors.
        """
        future.fix_prediction_target(
            factors=factors,
            key=self.pred_target,
            session_filter=lambda ts: is_cn_market_session(ts)['is_valid'],
            inplace=True,
            pred_length=15 * 60
        )

        future.wavelet_prediction_target(
            factors=factors,
            key=self.pred_target,
            session_filter=lambda ts: is_cn_market_session(ts)['is_valid'],
            inplace=True,
            decoder=self.decoder,
            decode_level=self.decoder.level
        )

        y = factors['target_smoothed'].to_numpy()
        return y

    def _cross_validation(self, x, y, factors: pd.DataFrame):
        """
        Performs cross-validation with linear regression.

        Args:
            factors (pd.DataFrame): DataFrame containing factors.

        Returns:
            Tuple: Cross-validation object and plotly figure.
        """

        x_axis = factors['market_time']

        # Drop rows with NaN or infinite values horizontally from x, y, and x_axis
        valid_mask = np.all(np.isfinite(x), axis=1) & np.isfinite(y)
        x = x[valid_mask]
        y = y[valid_mask]
        x_axis = x_axis[valid_mask]

        self.cv.validate(x=x, y=y)
        fig = self.cv.plot(x=x, y=y, x_axis=x_axis)
        fig = self.plot_synthetic(factors=factors, fig=fig)
        return fig

    def plot_synthetic(self, factors: pd.DataFrame, fig, plot_wavelet: bool = True):
        import plotly.graph_objects as go
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

        if plot_wavelet:
            for level in range(self.decoder.level + 1):
                local_extreme = self.decoder.local_extremes(ticker=self.pred_target, level=level)

                if not local_extreme:
                    break

                y, x, wave_flag = zip(*local_extreme)
                x = [datetime.datetime.fromtimestamp(_, tz=TIME_ZONE) for _ in x]

                trace = go.Scatter(x=x, y=y, mode='lines', name=f'decode level {level}', yaxis='y2')
                fig.add_trace(trace)

        fig.update_xaxes(
            rangebreaks=[dict(bounds=[0, 9.5], pattern="hour"), dict(bounds=[11.5, 13], pattern="hour"), dict(bounds=[15, 24], pattern="hour")],
        )
        fig.update_layout(
            yaxis2=dict(
                title="Synthetic",
                overlaying='y',
                side='right',
                showgrid=False
            )
        )

        return fig

    def _dump_result(self, market_date: datetime.date, factors: pd.DataFrame, fig):
        """
        Dumps the cross-validation results to CSV and HTML files.

        Args:
            market_date (datetime.date): Current market date.
            factors (pd.DataFrame): DataFrame containing factors.
            fig: Plotly figure from cross-validation.
        """
        dump_dir = f'Validation.{self.validation_id.split("-")[0]}'
        os.makedirs(dump_dir, exist_ok=True)

        entry_dir = pathlib.Path(dump_dir, f'{market_date:%Y-%m-%d}')
        os.makedirs(entry_dir, exist_ok=True)

        factors.to_csv(pathlib.Path(entry_dir, f'{self.factor.name}.validation.csv'))
        fig.write_html(pathlib.Path(entry_dir, f'{self.factor.name}.validation.html'))

    def validation(self, market_date: datetime.date):
        """
        Performs factor validation for the given market date.

        Args:
            market_date (datetime.date): Current market date.
        """
        if not self.factor_value:
            return

        LOGGER.info(f'{market_date} validation started with {len(self.factor_value):,} obs.')

        # Step 1: Add define prediction target
        factor_metrics = pd.DataFrame(self.factor_value).T

        x = self._define_inputs(factors=factor_metrics)
        y = self._define_prediction(factors=factor_metrics)

        # Step 2: Regression analysis
        fig = self._cross_validation(x=x, y=y, factors=factor_metrics)

        # Step 3: Dump the results
        self._dump_result(market_date=market_date, factors=factor_metrics, fig=fig)

    def _collect_synthetic(self, timestamp: float, current_bar: BarData | None, last_update: float, entry_log: dict[str, float]):
        """
        Collects synthetic index data.

        Args:
            timestamp (float): Current timestamp.
            current_bar (BarData): Current bar data.
            last_update (float): Last update timestamp.
            entry_log (dict): Dictionary to store collected data.

        Returns:
            BarData | None: Updated bar data.
        """
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
        """
        Collects factor data.

        Args:
            timestamp (float): Current timestamp.
            last_update (float): Last update timestamp.
            entry_log (dict): Dictionary to store collected data.
        """
        if timestamp >= last_update + self.sampling_interval:
            factors = collect_factor(monitors=self.factor)
            entry_log.update(factors)

    def _collect_market_price(self, ticker: str, market_price: float, entry_log: dict[str, float]):
        """
        Collects market price data.

        Args:
            ticker (str): Ticker symbol.
            market_price (float): Market price.
            entry_log (dict): Dictionary to store collected data.
        """
        synthetic_price = self.synthetic.index_price

        if entry_log is not None and (key := f'{ticker}.market_price') not in entry_log:
            entry_log[key] = market_price

        if entry_log is not None and (key := f'{self.synthetic.index_name}.market_price') not in entry_log:
            entry_log[key] = synthetic_price

    def run(self):
        """
        Runs the factor validation process.
        """
        self.init_factor()
        self.init_replay()

        last_update = 0.
        entry_log = None
        current_bar: BarData | None = None

        for market_data in self.replay:  # type: MarketData
            if not is_cn_market_session(market_data.timestamp)['is_valid']:
                continue

            MDS.on_market_data(market_data=market_data)

            timestamp = market_data.timestamp
            ticker = market_data.ticker
            market_price = market_data.market_price

            if timestamp >= last_update + self.sampling_interval:
                timestamp_index = timestamp // self.sampling_interval * self.sampling_interval
                self.factor_value[timestamp_index] = entry_log = {}

            current_bar = self._collect_synthetic(timestamp=timestamp, current_bar=current_bar, last_update=last_update, entry_log=entry_log)
            self._collect_factor(timestamp=timestamp, last_update=last_update, entry_log=entry_log)
            self._collect_market_price(ticker=ticker, market_price=market_price, entry_log=entry_log)

            last_update = timestamp // self.sampling_interval * self.sampling_interval


class FactorBatchValidation(FactorValidation):
    """
    Class for batch factor validation with multiple factors.

    Attributes:
        Same as FactorValidation, with additional attributes for multiple factors.

    Methods:
        init_factor(self, **kwargs) -> list[MarketDataMonitor]: Override to initialize multiple factors.
        _collect_factor(self, timestamp: float, last_update: float, entry_log: dict[str, float]): Override to collect data for multiple factors.
        reset(self): Reset multiple factors.
        _dump_result(self, market_date: datetime.date, factors: pd.DataFrame, fig): Override to dump results for multiple factors.
    """

    def __init__(self, **kwargs):
        """
        Initializes the FactorBatchValidation instance.

        Args:
            **kwargs: Additional parameters for configuration.
        """
        super().__init__(**kwargs)

        self.poly_degree = kwargs.get('poly_degree', 2)
        self.override_cache = kwargs.get('override_cache', False)

        self.features: list[str] = ['MACD.Index.Trigger.Synthetic', 'Entropy.Price.EMA']
        self.factor: list[MarketDataMonitor] = []

        self.factor_pool = factor_pool.FACTOR_POOL
        self.factor_cache: MarketDataMonitor | None = None

    def init_factor(self, **kwargs) -> list[MarketDataMonitor]:
        """
        Initializes multiple factors for validation.

        Args:
            **kwargs: Additional parameters for factor configuration.

        Returns:
            list[MarketDataMonitor]: Initialized list of market data monitors.
        """
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

        self.factor_cache = factor_pool.FactorPoolDummyMonitor(factor_pool=self.factor_pool)

        for _ in self.factor:
            MDS.add_monitor(_)

        MDS.add_monitor(self.synthetic)

        if not self.override_cache:
            MDS.add_monitor(self.factor_cache)

        return self.factor

    def init_cache(self, market_date: datetime.date):
        self.factor_pool.load(market_date=market_date)
        factor_existed = self.factor_pool.factor_names(market_date=market_date)

        for factor in self.factor:
            factor_prefix = factor.name.removeprefix('Monitor.')

            for _ in factor_existed:
                if factor_prefix in _:
                    factor.enabled = False
                    LOGGER.info(f'Factor {factor.name} found in the factor cache, and will be disabled.')
                    break

    def update_cache(self, market_date: datetime):
        if self.override_cache:
            LOGGER.info('Cache overridden!')
            self.factor_pool.batch_update(factors=self.factor_value)
        else:
            LOGGER.info('Cache updated!')
            exclude_keys = self.factor_pool.factor_names(market_date=market_date)
            self.factor_pool.batch_update(factors=self.factor_value, exclude_keys=exclude_keys)
        self.factor_pool.dump()

    def bod(self, market_date: datetime.date, **kwargs) -> None:

        if not self.override_cache:
            self.init_cache(market_date=market_date)

        super().bod(market_date=market_date, **kwargs)

        # no replay task is needed, remove all tasks
        if not self.override_cache:
            if all([not factor.enabled for factor in self.factor]):
                self.replay.replay_subscription.clear()
                self.subscription.clear()  # need to ensure the synchronization of the subscription
                LOGGER.info(f'{market_date} All factor is cached, skip this day.')
                self.factor_value.update(self.factor_pool.storage[market_date])

    def eod(self, market_date: datetime.date, **kwargs) -> None:
        self.update_cache(market_date=market_date)
        super().eod(market_date=market_date, **kwargs)

    def _collect_factor(self, timestamp: float, last_update: float, entry_log: dict[str, float]):
        super()._collect_factor(timestamp=timestamp, last_update=last_update, entry_log=entry_log)

        if not self.override_cache:
            factors = collect_factor(monitors=self.factor_cache)
            entry_log.update(factors)

    def _dump_result(self, market_date: datetime.date, factors: pd.DataFrame, fig):
        """
        Dumps results for multiple factors.

        Args:
            market_date (datetime.date): Current market date.
            factors (pd.DataFrame): DataFrame containing factors.
            fig: Plotly figure from cross-validation.
        """
        dump_dir = f'BatchValidation.{self.validation_id.split("-")[0]}'
        os.makedirs(dump_dir, exist_ok=True)

        entry_dir = pathlib.Path(dump_dir, f'{market_date:%Y-%m-%d}')
        os.makedirs(entry_dir, exist_ok=True)

        file_name = f'{"".join([f"[{factor.name}]" for factor in self.factor])}.validation'
        factors.to_csv(pathlib.Path(entry_dir, f'{file_name}.csv'))
        fig.write_html(pathlib.Path(entry_dir, f'{file_name}.html'), include_plotlyjs='cdn')

    def reset(self):
        """
        Resets multiple factors.
        """
        for _ in self.factor:
            _.clear()
            _.enabled = True

        self.synthetic.clear()
        self.factor_value.clear()
        self.factor_cache.clear()
        self.decoder.clear()

    def _define_inputs(self, factors: pd.DataFrame):
        """
        Defines input features for regression analysis.

        Args:
            factors (pd.DataFrame): DataFrame containing factors.
        """
        factors['market_time'] = [datetime.datetime.fromtimestamp(_) for _ in factors.index]
        factors['bias'] = 1.

        session_dummies(timestamp=factors.index, inplace=factors)

        # default features
        features = {'Dummies.IsOpening', 'Dummies.IsClosing'}
        features.update(self.features)
        feature_original = factors[list(features)]
        x_matrix = factors.loc[:, list(features)]

        for i in range(1, self.poly_degree):
            additional_feature = poly_kernel(feature_original, degree=i + 1)

            for _ in additional_feature:
                x_matrix[_] = additional_feature[_]

        invalid_factor: pd.DataFrame = x_matrix.loc[:, x_matrix.nunique() == 1]
        if not invalid_factor.empty:
            LOGGER.error(f'Invalid factor {invalid_factor}, add epsilon to avoid multicolinearity')
            for name in invalid_factor.columns:
                x_matrix[name] = 1 + np.random.normal(scale=0.1, size=len(x_matrix))

        x_matrix['bias'] = 1.

        for _ in x_matrix.columns:
            self.coefficients[_] = 0.

        x = x_matrix.to_numpy()

        return x


class InterTemporalValidation(FactorBatchValidation):
    """
    model is trained prior to the beginning of the day, using multi-day data
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.training_days: int = kwargs.get('training_days', 5)
        self.factor_value_storage = deque(maxlen=self.training_days)

    def validation(self, market_date: datetime.date):
        if not self.factor_value_storage:
            self.factor_value_storage.append(self.factor_value.copy())
            return

        LOGGER.info(f'{market_date} validation started with {len(self.factor_value_storage):,} days obs.')

        # Step 1: define training set
        x_train, y_train = [], []
        for factor_value in self.factor_value_storage:
            factor_metrics = pd.DataFrame(factor_value).T

            _x = self._define_inputs(factors=factor_metrics)
            _y = self._define_prediction(factors=factor_metrics)

            x_train.append(_x)
            y_train.append(_y)
            self.decoder.clear()

        x_train = np.concatenate(x_train)
        y_train = np.concatenate(y_train)

        # Step 2: define validation set
        factor_metrics = pd.DataFrame(self.factor_value).T
        x_val = self._define_inputs(factors=factor_metrics)
        y_val = self._define_prediction(factors=factor_metrics)

        fig = self._out_sample_validation(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, factors=factor_metrics)

        # Step 3: Dump the results
        self._dump_result(market_date=market_date, factors=factor_metrics, fig=fig)

        # step 4: store factor value
        self.factor_value_storage.append(self.factor_value.copy())

    def _out_sample_validation(self, x_train, y_train, x_val, y_val, factors: pd.DataFrame):
        x_axis = factors['market_time']

        valid_mask = np.all(np.isfinite(x_train), axis=1) & np.isfinite(y_train)
        x_train = x_train[valid_mask]
        y_train = y_train[valid_mask]

        valid_mask = np.all(np.isfinite(x_val), axis=1) & np.isfinite(y_val)
        x_val = x_val[valid_mask]
        y_val = y_val[valid_mask]
        x_axis = x_axis[valid_mask]

        self.cv.validate_out_sample(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
        fig = self.cv.plot(x_axis=x_axis)
        fig = self.plot_synthetic(factors=factors, fig=fig)
        return fig

    def _dump_result(self, market_date: datetime.date, factors: pd.DataFrame, fig):
        dump_dir = f'InterTemporalValidation.{self.validation_id.split("-")[0]}'
        os.makedirs(dump_dir, exist_ok=True)

        entry_dir = pathlib.Path(dump_dir, f'{market_date:%Y-%m-%d}')
        os.makedirs(entry_dir, exist_ok=True)

        if len(self.factor) > 1:
            file_name = f'InterTemporal.validation'
        else:
            file_name = f'{"".join([f"[{factor.name}]" for factor in self.factor])}.validation'

        factors.to_csv(pathlib.Path(entry_dir, f'{file_name}.factors.csv'))
        fig.write_html(pathlib.Path(entry_dir, f'{file_name}.validation.html'), include_plotlyjs='cdn')
        self.model.dump(pathlib.Path(entry_dir, f'{file_name}.model.json'))
        self.cv.metrics.to_html(pathlib.Path(entry_dir, f'{file_name}.metrics.html'))


def main():
    """
    Main function to run factor validation or batch validation.
    """
    # fv = FactorValidation()
    # fv = FactorBatchValidation()
    fv = InterTemporalValidation()
    fv.run()
    safe_exit()


if __name__ == '__main__':
    main()
