"""
This script is designed for factor validation using linear regression.
"""
__package__ = 'Quark.Factor'

import datetime
import os
import pathlib
import shutil
import uuid
from collections import deque

import numpy as np
import pandas as pd
from AlgoEngine.Engine import ProgressiveReplay
from PyQuantKit import MarketData, BarData

from . import LOGGER, MDS, IndexWeight, ALPHA_0001, ALPHA_05, collect_factor, FactorMonitor
from .Correlation import *
from .Distribution import *
from .LowPass import *
from .Misc import SyntheticIndexMonitor
from .TradeFlow import *
from .decoder import RecursiveDecoder
from .factor_pool import FACTOR_POOL, FactorPoolDummyMonitor
from ..API import historical
from ..Backtest import simulated_env
from ..Base import safe_exit, GlobalStatics
from ..Calibration.Boosting.xgboost import *
from ..Calibration.Linear.bootstrap import *
from ..Calibration.cross_validation import CrossValidation
from ..Calibration.dummies import is_market_session
from ..DataLore.utils import define_inputs, define_prediction
from ..Misc import helper
from ..Profile import cn

cn.profile_cn_override()
LOGGER = LOGGER.getChild('validation')
DUMMY_WEIGHT = False
TIME_ZONE = GlobalStatics.TIME_ZONE
RANGE_BREAK = GlobalStatics.RANGE_BREAK


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
        self.pred_target = f'{self.index_name}.market_price'
        self.features = ['MACD.Index.Trigger.Synthetic']

        self.factor: FactorMonitor | None = None
        self.synthetic = SyntheticIndexMonitor(index_name=self.index_name, weights=self.index_weights)
        self.subscription = set()
        self.replay: ProgressiveReplay | None = None
        self.factor_value: dict[float, dict[str, float]] = {}
        self.metrics = {}

        self.model = RidgeRegression(alpha=1.0)  # for ridge regression (build linear baseline)
        self.cv = CrossValidation(model=self.model, folds=10, shuffle=True, strict_no_future=True)

    def init_factor(self, **kwargs) -> FactorMonitor:
        """
        Initializes the factor for validation.

        Args:
            **kwargs: Additional parameters for factor configuration.

        Returns:
            MarketDataMonitor: Initialized market data monitor.
        """
        self.factor = DivergenceIndexAdaptiveMonitor(
            weights=self.index_weights,
            sampling_interval=15,
            baseline_window=20
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
        index_weights = IndexWeight(
            index_name=self.index_name,
            **helper.load_dict(
                file_path=pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res', f'index_weights.{self.index_name}.{market_date:%Y%m%d}.json'),
                json_dict=simulated_env.query(ticker=self.index_name, market_date=market_date, topic='index_weights')
            )
        )

        # A lite setting for fast debugging
        if DUMMY_WEIGHT:
            for _ in list(index_weights.keys())[10:]:
                index_weights.pop(_)

        # Step 0: Update index weights
        self.index_weights.update(index_weights)
        self.index_weights.normalize()
        self.synthetic.weights = self.index_weights

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

        self.init_factor()

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
        MDS.clear()

        self.factor_value.clear()
        self.decoder.clear()
        self.cv.clear()

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
        factors['market_time'] = [datetime.datetime.fromtimestamp(_, tz=TIME_ZONE) for _ in factors.index]
        factors['bias'] = 1.

        x = define_inputs(factor_value=factors, input_vars=self.features, poly_degree=1).to_numpy()

        return x

    def _define_prediction(self, factors: pd.DataFrame):
        y = define_prediction(factor_value=factors, pred_var='target_smoothed', decoder=self.decoder, key=f'{self.synthetic.name.removeprefix("Monitor.")}.market_price')
        y = y.to_numpy()
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

        self.cv.cross_validate(x=x, y=y)
        self.cv.x_axis = x_axis

    def _plot_cv(self, factors: pd.DataFrame, plot_wavelet: bool = True):
        import plotly.graph_objects as go
        fig = self.cv.plot()

        candlestick_trace = go.Candlestick(
            name='Synthetic',
            x=factors['market_time'],
            open=factors[f'{self.synthetic.index_name}.open_price'],
            high=factors[f'{self.synthetic.index_name}.high_price'],
            low=factors[f'{self.synthetic.index_name}.low_price'],
            close=factors[f'{self.synthetic.index_name}.close_price'],
            yaxis='y3'
        )
        fig.add_trace(candlestick_trace)

        if plot_wavelet:
            for level in range(self.decoder.level + 1):
                local_extreme = self.decoder.local_extremes(ticker=self.pred_target, level=level)

                if not local_extreme:
                    break

                y, x, wave_flag = zip(*local_extreme)
                x = [datetime.datetime.fromtimestamp(_, tz=TIME_ZONE) for _ in x]

                trace = go.Scatter(x=x, y=y, mode='lines', name=f'decode level {level}', yaxis='y3')
                fig.add_trace(trace)

        fig.update_xaxes(
            rangebreaks=RANGE_BREAK,
        )

        fig.update_layout(
            yaxis3=dict(
                title="Synthetic",
                anchor="x",
                overlaying='y',
                side='right',
                showgrid=False
            )
        )

        return fig

    def _dump_result(self, market_date: datetime.date, factors: pd.DataFrame):
        """
        Dumps the cross-validation results to CSV and HTML files.

        Args:
            market_date (datetime.date): Current market date.
            factors (pd.DataFrame): DataFrame containing factors.
        """
        dump_dir = f'Validation.{self.validation_id.split("-")[0]}'
        os.makedirs(dump_dir, exist_ok=True)

        entry_dir = pathlib.Path(dump_dir, f'{market_date:%Y-%m-%d}')
        os.makedirs(entry_dir, exist_ok=True)

        factors.to_csv(pathlib.Path(entry_dir, f'{self.factor.name}.validation.csv'))
        fig = self._plot_cv(factors=factors)
        fig.write_html(pathlib.Path(entry_dir, f'{self.factor.name}.validation.html'))

        self.metrics[market_date] = self.cv.metrics.metrics
        pd.DataFrame(self.metrics).T.to_csv(pathlib.Path(dump_dir, f'metrics.csv'))

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
        self._cross_validation(x=x, y=y, factors=factor_metrics)

        # Step 3: Dump the results
        self._dump_result(market_date=market_date, factors=factor_metrics)

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
            timestamp_index = (timestamp // self.sampling_interval) * self.sampling_interval

            if current_bar is not None:
                entry_log[f'{self.synthetic.index_name}.open_price'] = current_bar.open_price
                entry_log[f'{self.synthetic.index_name}.close_price'] = current_bar.close_price
                entry_log[f'{self.synthetic.index_name}.high_price'] = current_bar.high_price
                entry_log[f'{self.synthetic.index_name}.low_price'] = current_bar.low_price
                entry_log[f'{self.synthetic.index_name}.notional'] = current_bar.notional

            current_bar = BarData(
                ticker='Synthetic',
                bar_start_time=datetime.datetime.fromtimestamp(timestamp_index, tz=TIME_ZONE),
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

        if entry_log is not None and (key := f'{ticker}.market_price') not in entry_log:
            entry_log[key] = market_price

        if entry_log is not None and f'{self.synthetic.index_name}.market_price' not in entry_log:
            entry_log.update(collect_factor(self.synthetic))

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
            if not is_market_session(market_data.timestamp):
                continue

            MDS.on_market_data(market_data=market_data)

            timestamp = market_data.timestamp
            ticker = market_data.ticker
            market_price = market_data.market_price

            if timestamp >= last_update + self.sampling_interval:
                timestamp_index = (timestamp // self.sampling_interval) * self.sampling_interval
                self.factor_value[timestamp_index] = entry_log = {}

            current_bar = self._collect_synthetic(timestamp=timestamp, current_bar=current_bar, last_update=last_update, entry_log=entry_log)
            self._collect_factor(timestamp=timestamp, last_update=last_update, entry_log=entry_log)
            self._collect_market_price(ticker=ticker, market_price=market_price, entry_log=entry_log)

            last_update = (timestamp // self.sampling_interval) * self.sampling_interval


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
        self.cache_dir = None

        self.features: list[str] = [
            'Skewness.PricePct.Index.Adaptive.Index',
            'Skewness.PricePct.Index.Adaptive.Slope',
            'Gini.PricePct.Index.Adaptive',
            'Coherence.Price.Adaptive.up',
            'Coherence.Price.Adaptive.down',
            'Coherence.Price.Adaptive.ratio',
            'Coherence.Volume.up',
            'Coherence.Volume.down',
            'Entropy.Price.Adaptive',
            'Entropy.Price',
            'Entropy.PricePct.Adaptive',
            'EMA.Divergence.Index.Adaptive.Index',
            'EMA.Divergence.Index.Adaptive.Diff',
            'EMA.Divergence.Index.Adaptive.Diff.EMA',
            'TradeFlow.EMA.Index',
            'Aggressiveness.EMA.Index',
        ]
        self.factor: list[FactorMonitor] = []

        self.factor_pool = FACTOR_POOL
        self.factor_cache = FactorPoolDummyMonitor(factor_pool=self.factor_pool)

    def init_factor(self, **kwargs) -> list[FactorMonitor]:
        """
        Initializes multiple factors for validation.

        Args:
            **kwargs: Additional parameters for factor configuration.

        Returns:
            list[MarketDataMonitor]: Initialized list of market data monitors.
        """
        self.factor = [
            CoherenceAdaptiveMonitor(
                sampling_interval=15,
                sample_size=20,
                baseline_window=100,
                weights=self.index_weights,
                center_mode='median',
                aligned_interval=True
            ),
            TradeCoherenceMonitor(
                sampling_interval=15,
                sample_size=20,
                weights=self.index_weights
            ),
            EntropyMonitor(
                sampling_interval=15,
                sample_size=20,
                weights=self.index_weights
            ),
            EntropyAdaptiveMonitor(
                sampling_interval=15,
                sample_size=20,
                weights=self.index_weights
            ),
            EntropyAdaptiveMonitor(
                sampling_interval=15,
                sample_size=20,
                weights=self.index_weights,
                ignore_primary=False,
                pct_change=True,
                name='Monitor.Entropy.PricePct.Adaptive'
            ),
            GiniIndexAdaptiveMonitor(
                sampling_interval=3 * 5,
                sample_size=20,
                baseline_window=100,
                weights=self.index_weights
            ),
            SkewnessIndexAdaptiveMonitor(
                sampling_interval=3 * 5,
                sample_size=20,
                baseline_window=100,
                weights=self.index_weights,
                aligned_interval=False
            ),
            DivergenceIndexAdaptiveMonitor(
                weights=self.index_weights,
                sampling_interval=15,
                baseline_window=20,
            ),
            TradeFlowEMAMonitor(
                discount_interval=1,
                alpha=ALPHA_05,
                weights=self.index_weights
            ),
            AggressivenessEMAMonitor(
                discount_interval=1,
                alpha=ALPHA_0001,
                weights=self.index_weights
            )
        ]

        for _ in self.factor:
            MDS.add_monitor(_)

        MDS.add_monitor(self.synthetic)

        if not self.override_cache:
            MDS.add_monitor(self.factor_cache)

        return self.factor

    def init_cache(self, market_date: datetime.date):
        self.factor_pool.load(market_date=market_date, factor_dir=self.cache_dir)
        factor_existed = self.factor_pool.factor_names(market_date=market_date)

        for factor in self.factor:
            factor_names = factor.factor_names(subscription=list(self.subscription))

            if all([_ in factor_existed for _ in factor_names]):
                factor.enabled = False
                LOGGER.info(f'Factor {factor.name} found in the factor cache, and will be disabled.')

    def update_cache(self, market_date: datetime):
        if self.override_cache:
            LOGGER.info('Cache overridden!')
            self.factor_pool.batch_update(factors=self.factor_value)
        else:
            exclude_keys = self.factor_pool.factor_names(market_date=market_date)
            self.factor_pool.batch_update(factors=self.factor_value, exclude_keys=exclude_keys)

            if all([name in exclude_keys for name in pd.DataFrame(self.factor_value).T.columns]):
                return

            LOGGER.info('Cache updated!')

        self.factor_pool.dump(factor_dir=self.cache_dir)

    def bod(self, market_date: datetime.date, **kwargs) -> None:

        super().bod(market_date=market_date, **kwargs)

        self.init_factor()

        if not self.override_cache:
            self.init_cache(market_date=market_date)

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
        # only un-cached monitors is registered
        super()._collect_factor(timestamp=timestamp, last_update=last_update, entry_log=entry_log)

        # collect cached monitors
        if (not self.override_cache) and timestamp >= last_update + self.sampling_interval:
            factors = collect_factor(monitors=self.factor_cache)
            entry_log.update(factors)

    def _plot_factors(self, factors: pd.DataFrame, precision=4):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Select relevant columns from factors
        selected_factors = factors[['market_time'] + self.features]
        hover_data = factors[self.features].astype(np.float64)

        # Create subplot
        fig = make_subplots(
            rows=len(self.features) + 1,
            cols=1,
            shared_xaxes=False,
            subplot_titles=['Synthetic'] + self.features,
            row_heights=[3] + [1] * len(self.features)
        )

        candlestick_trace = go.Candlestick(
            name='Synthetic',
            x=factors['market_time'],
            open=factors[f'{self.synthetic.index_name}.open_price'],
            high=factors[f'{self.synthetic.index_name}.high_price'],
            low=factors[f'{self.synthetic.index_name}.low_price'],
            close=factors[f'{self.synthetic.index_name}.close_price'],
            showlegend=True,
        )
        fig.add_trace(candlestick_trace, row=1, col=1)
        fig['layout'][f'yaxis']['title'] = 'Synthetic'

        # Add traces for each feature
        for i, feature in enumerate(self.features):
            trace = go.Scatter(
                x=selected_factors['market_time'],
                y=selected_factors[feature],
                mode='lines',
                name=feature,
                customdata=hover_data,
                hovertemplate='<br>'.join(
                    ['Datetime: %{x:%Y-%m-%d:%h}'] +
                    ['<b>' + feature + '</b><b>' + f": %{{y:.{precision}f}}" + '</b>'] +
                    [self.features[_] + f": %{{customdata[{_}]:.{precision}f}}" for _ in range(len(self.features)) if _ != i] +
                    ['<extra></extra>']  # otherwise another legend will be shown
                ),
                showlegend=True,
            )

            fig.add_trace(trace, row=i + 2, col=1)
            # fig['layout'][f'yaxis{i + 2}']['title'] = feature
            fig.update_layout(
                {
                    f'yaxis{i + 2}': dict(
                        title=feature,
                        showgrid=True,
                        zeroline=True,
                        showticklabels=True,
                        showspikes=True,
                        # spikemode='across',
                        spikesnap='cursor',
                        spikethickness=-2,
                        # showline=False,
                        # spikedash='solid'
                    )
                }
            )

        fig.update_layout(
            title=dict(text="Factor Values for Synthetic"),
            height=200 * (3 + len(self.features)),
            template='simple_white',
            # legend_tracegroupgap=330,
            hovermode='x unified',
            legend_traceorder="normal"
        )

        fig.update_traces(xaxis=f'x1')

        fig.update_xaxes(
            tickformat='%H:%M:%S',
            gridcolor='black',
            griddash='dash',
            minor_griddash="dot",
            showgrid=True,
            spikethickness=-2,
            rangebreaks=RANGE_BREAK,
            rangeslider_visible=False
        )

        return fig

    def _dump_result(self, market_date: datetime.date, factors: pd.DataFrame):
        """
        Dumps results for multiple factors.

        Args:
            market_date (datetime.date): Current market date.
            factors (pd.DataFrame): DataFrame containing factors.
        """
        dump_dir = f'{self.__class__.__name__}.{self.model.__class__.__name__}.{self.validation_id.split("-")[0]}'
        os.makedirs(dump_dir, exist_ok=True)

        entry_dir = pathlib.Path(dump_dir, f'{market_date:%Y-%m-%d}')
        os.makedirs(entry_dir, exist_ok=True)

        if len(self.factor) > 2:
            file_name = f'{self.__class__.__name__}'
        else:
            file_name = f'{"".join([f"[{factor.name}]" for factor in self.factor])}.validation'

        if self.cv.x_val is not None:
            fig = self._plot_cv(factors=factors)
            fig.write_html(pathlib.Path(entry_dir, f'{file_name}.pred.html'))

            self.model.dump(pathlib.Path(entry_dir, f'{file_name}.model.json'))
            self.cv.metrics.to_html(pathlib.Path(entry_dir, f'{file_name}.metrics.html'))

            self.metrics[market_date] = self.cv.metrics.metrics
            pd.DataFrame(self.metrics).T.to_csv(pathlib.Path(dump_dir, f'metrics.csv'))

        factors.to_csv(pathlib.Path(entry_dir, f'{file_name}.factors.csv'))

        fig = self._plot_factors(factors=factors)
        fig.write_html(pathlib.Path(entry_dir, f'{file_name}.factor.html'))

    def reset(self):
        """
        Resets multiple factors.
        """

        MDS.clear()
        self.factor.clear()

        self.synthetic.clear()
        self.factor_value.clear()
        self.factor_cache.clear()
        self.decoder.clear()
        self.cv.clear()

    def _define_inputs(self, factors: pd.DataFrame):
        """
        Defines input features for regression analysis.

        Args:
            factors (pd.DataFrame): DataFrame containing factors.
        """
        factors['market_time'] = [datetime.datetime.fromtimestamp(_, tz=TIME_ZONE) for _ in factors.index]
        factors['bias'] = 1.

        x = define_inputs(factor_value=factors, input_vars=self.features, poly_degree=self.poly_degree).to_numpy()
        return x

    def _define_prediction(self, factors: pd.DataFrame):
        y = super()._define_prediction(factors=factors)

        # multi factors regression should adjust the baseline
        y = y - np.nanmedian(y)
        return y


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
            factor_metrics = pd.DataFrame(self.factor_value).T
            _ = self._define_inputs(factors=factor_metrics)
            _ = self._define_prediction(factors=factor_metrics)
            self._dump_result(market_date=market_date, factors=factor_metrics)
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

        self._out_sample_validation(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, factors=factor_metrics)

        # Step 3: Dump the results
        self._dump_result(market_date=market_date, factors=factor_metrics)

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

        # if isinstance(self.model, RidgeRegression):
        #     self.model.optimal_alpha(x=x_train, y=y_train)

        self.cv.validate(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
        self.cv.x_axis = x_axis


class HybridValidation(InterTemporalValidation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.model = RandomForest()
        self.model = XGBoost()
        self.cv.model = self.model
        self.poly_degree = kwargs.get('poly_degree', 2)

    # def _define_prediction(self, factors: pd.DataFrame):
    #     y = FactorValidation._define_prediction(self, factors=factors)
    #     return y


class FactorValidatorExperiment(InterTemporalValidation):
    """
    this validator is designed for experiments
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = RidgeRegression(alpha=0.1)
        # self.model = XGBoost()
        self.cv = CrossValidation(model=self.model, folds=10, shuffle=True, strict_no_future=True)
        self.features: list[str] = [
            'Skewness.PricePct.Index.Adaptive.Index',
            'Skewness.PricePct.Index.Adaptive.Slope',
            'Gini.PricePct.Index.Adaptive',
            'Coherence.Price.Adaptive.up',
            'Coherence.Price.Adaptive.down',
            'Coherence.Price.Adaptive.ratio',
            'Coherence.Volume.up',
            'Coherence.Volume.down',
            'Entropy.Price.Adaptive',
            'Entropy.Price',
            'Entropy.PricePct.Adaptive',
            'EMA.Divergence.Index.Adaptive.Index',
            'EMA.Divergence.Index.Adaptive.Diff',
            'EMA.Divergence.Index.Adaptive.Diff.EMA',
            # 'Trade.Clustering.Index.Adaptive.Index',
            # 'Trade.Clustering.Index.Adaptive.Weighted',
        ]

        self.cache_dir = kwargs.get('cache_dir', pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res', 'tmp_factor_cache'))

        # experimenting with the factors requires clearing caches regularly
        if os.path.isdir(self.cache_dir) and self.override_cache:
            shutil.rmtree(self.cache_dir)  # remove dir and all contains
            LOGGER.info(f'Factor cache {self.cache_dir} removed!')

        validation_id = 1

        while True:
            dump_dir = f'{self.__class__.__name__}.{self.model.__class__.__name__}.{validation_id}'
            if os.path.isdir(dump_dir):
                validation_id += 1
            else:
                break

        self.validation_id = kwargs.get('validation_id', f'{validation_id}-val')

    def init_factor(self, **kwargs) -> list[FactorMonitor]:
        """
        Initializes multiple factors for validation.

        Args:
            **kwargs: Additional parameters for factor configuration.

        Returns:
            list[MarketDataMonitor]: Initialized list of market data monitors.
        """
        self.factor = [
            CoherenceAdaptiveMonitor(
                sampling_interval=15,
                sample_size=20,
                baseline_window=100,
                weights=self.index_weights,
                center_mode='median',
                aligned_interval=True
            ),
            TradeCoherenceMonitor(
                sampling_interval=15,
                sample_size=20,
                weights=self.index_weights
            ),
            EntropyMonitor(
                sampling_interval=15,
                sample_size=20,
                weights=self.index_weights
            ),
            EntropyAdaptiveMonitor(
                sampling_interval=15,
                sample_size=20,
                weights=self.index_weights
            ),
            EntropyAdaptiveMonitor(
                sampling_interval=15,
                sample_size=20,
                weights=self.index_weights,
                ignore_primary=False,
                pct_change=True,
                name='Monitor.Entropy.PricePct.Adaptive'
            ),
            GiniIndexAdaptiveMonitor(
                sampling_interval=3 * 5,
                sample_size=20,
                baseline_window=100,
                weights=self.index_weights
            ),
            SkewnessIndexAdaptiveMonitor(
                sampling_interval=3 * 5,
                sample_size=20,
                baseline_window=100,
                weights=self.index_weights,
                aligned_interval=False
            ),
            DivergenceIndexAdaptiveMonitor(
                weights=self.index_weights,
                sampling_interval=15,
                baseline_window=20,
            ),
            # TradeClusteringIndexAdaptiveMonitor(
            #     weights=self.index_weights,
            #     sampling_interval=15,
            #     sample_size=20,
            #     baseline_window=100
            # )
        ]

        self.factor_cache = FactorPoolDummyMonitor(factor_pool=self.factor_pool)

        for _ in self.factor:
            MDS.add_monitor(_)

        MDS.add_monitor(self.synthetic)

        if not self.override_cache:
            MDS.add_monitor(self.factor_cache)

        return self.factor

    def update_cache(self, market_date: datetime):
        self.factor_pool.batch_update(factors=self.factor_value)
        self.factor_pool.dump(factor_dir=self.cache_dir)


def main():
    """
    Main function to run factor validation or batch validation.
    """

    # validator = FactorValidation()
    # validator = FactorBatchValidation()
    validator = InterTemporalValidation()
    # validator = HybridValidation()
    # validator = FactorValidatorExperiment(override_cache=True)
    validator.run()
    safe_exit()


if __name__ == '__main__':
    main()
