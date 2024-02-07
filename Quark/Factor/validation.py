"""
This script is designed for factor validation using linear regression.
"""
__package__ = 'Quark.Factor'

import datetime
import os
import pathlib
import random
import shutil
from collections import deque

import numpy as np
import pandas as pd
from AlgoEngine.Engine import ProgressiveReplay
from PyQuantKit import MarketData

from . import LOGGER, MDS, IndexWeight, collect_factor, FactorMonitor
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
START_DATE = datetime.date(2023, 1, 1)
END_DATE = datetime.date(2023, 4, 1)


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
        # Params for index
        self.index_name = kwargs.get('index_name', '000016.SH')
        self.index_weights = IndexWeight(index_name='000016.SH')

        # Params for replay
        self.dtype = kwargs.get('dtype', 'TradeData')
        self.start_date = kwargs.get('start_date', START_DATE)
        self.end_date = kwargs.get('end_date', END_DATE)

        # Params for sampling
        self.sampling_interval = kwargs.get('sampling_interval', 10.)

        # Params for validation
        self.poly_degree = kwargs.get('poly_degree', 1)
        self.pred_var = kwargs.get('pred_var', ['target_smoothed', 'target_actual'])
        self.decoder = RecursiveDecoder(level=3)
        self.pred_target = f'{self.index_name}.market_price'
        self.features = [
            'TradeFlow.Adaptive.Index.Imbalance',
            'TradeFlow.Adaptive.Index.Entropy',
            'TradeFlow.Adaptive.Index.Boosted',
            'TradeFlow.Adaptive.Index.Slope',
        ]

        self.factor: FactorMonitor | None = None
        self.synthetic = SyntheticIndexMonitor(index_name=self.index_name, weights=self.index_weights, interval=self.sampling_interval)
        self.subscription = set()
        self.factor_value: dict[float, dict[str, float]] = {}

        self.model = {pred_var: RidgeRegression(alpha=1., exponential_decay=0.25, fixed_decay=0.5) for pred_var in self.pred_var}  # for ridge regression (build linear baseline)
        self.cv = {pred_var: CrossValidation(model=self.model[pred_var], folds=10, shuffle=True, strict_no_future=True) for pred_var in self.pred_var}
        self.metrics = {pred_var: {} for pred_var in self.pred_var}
        self.validation_id = kwargs.get('validation_id', self._get_validation_id())

    def _get_validation_id(self):
        validation_id = 1

        while True:
            dump_dir = f'{self.__class__.__name__}.{validation_id}'
            if os.path.isdir(dump_dir):
                validation_id += 1
            else:
                break

        return validation_id

    def _collect_factor(self, entry_log: dict[str, float]):
        factors = collect_factor(monitors=self.factor)
        entry_log.update(factors)

        synthetic = collect_factor(monitors=self.synthetic)
        entry_log.update(synthetic)

    def _cross_validation(self, x, y, factors: pd.DataFrame, cv: CrossValidation):
        valid_mask = np.all(np.isfinite(x), axis=1) & np.isfinite(y)
        x = x[valid_mask]
        y = y[valid_mask]
        x_axis = np.array([datetime.datetime.fromtimestamp(_, tz=TIME_ZONE) for _ in factors.index])[valid_mask]

        cv.cross_validate(x=x, y=y)
        cv.x_axis = x_axis

    def _candle_sticks(self, factor_value: pd.DataFrame):
        import plotly.graph_objects as go

        candlestick_trace = go.Candlestick(
            name='Synthetic',
            x=[datetime.datetime.fromtimestamp(_, tz=TIME_ZONE) for _ in factor_value.index],
            open=factor_value[f'{self.synthetic.index_name}.open_price'],
            high=factor_value[f'{self.synthetic.index_name}.high_price'],
            low=factor_value[f'{self.synthetic.index_name}.low_price'],
            close=factor_value[f'{self.synthetic.index_name}.close_price'],
        )

        return candlestick_trace

    def _plot_cv(self, pred_var: str, factors: pd.DataFrame, plot_wavelet: bool = True):
        import plotly.graph_objects as go
        fig = self.cv[pred_var].plot()

        candlestick_trace = self._candle_sticks(factor_value=factors)
        candlestick_trace['yaxis'] = 'y3'
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
            tickformat='%H:%M:%S',
            gridcolor='black',
            griddash='dash',
            minor_griddash="dot",
            showgrid=True,
            spikethickness=-2,
            rangebreaks=RANGE_BREAK,
            rangeslider_visible=False
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

    def _update_subscription(self, replay: ProgressiveReplay):
        """
        Updates market data subscriptions based on index weights.
        """
        subscription = set(self.index_weights.keys())

        for _ in subscription:
            if _ not in self.subscription:
                replay.add_subscription(ticker=_, dtype='TradeData')

        for _ in self.subscription:
            if _ not in subscription:
                replay.remove_subscription(ticker=_, dtype='TradeData')

        self.subscription.update(subscription)

    def initialize_factor(self, **kwargs) -> FactorMonitor:
        """
        Initializes the factor for validation.

        Args:
            **kwargs: Additional parameters for factor configuration.

        Returns:
            MarketDataMonitor: Initialized market data monitor.
        """
        self.factor = TradeFlowAdaptiveIndexMonitor(
            sampling_interval=3 * 5,
            sample_size=20,
            baseline_window=100,
            weights=self.index_weights,
            aligned_interval=False
        )

        MDS.add_monitor(self.factor)
        MDS.add_monitor(self.synthetic)

        return self.factor

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
        factor_value = pd.DataFrame(self.factor_value).T

        for pred_var in self.pred_var:
            cv = self.cv[pred_var]

            x = define_inputs(factor_value=factor_value, input_vars=self.features, poly_degree=self.poly_degree).to_numpy()
            y = define_prediction(factor_value=factor_value, pred_var=pred_var, decoder=self.decoder, key=self.pred_target)
            y = (y - np.nanmedian(y)).to_numpy()

            # Step 2: Regression analysis
            self._cross_validation(x=x, y=y, factors=factor_value, cv=cv)

    def dump_result(self, market_date: datetime.date):
        dump_dir = f'{self.__class__.__name__}.{self.validation_id}'
        os.makedirs(dump_dir, exist_ok=True)

        factor_value = pd.DataFrame(self.factor_value).T

        entry_dir = pathlib.Path(dump_dir, f'{market_date:%Y-%m-%d}')
        os.makedirs(entry_dir, exist_ok=True)

        factor_value.to_csv(pathlib.Path(entry_dir, f'{self.factor.name}.validation.csv'))
        for pred_var in self.pred_var:
            model = self.model[pred_var]
            cv = self.cv[pred_var]
            metrics = self.metrics[pred_var]

            fig = self._plot_cv(pred_var=pred_var, factors=factor_value)
            fig.write_html(pathlib.Path(entry_dir, f'{self.factor.name}.{pred_var}.validation.html'))

            model.dump(pathlib.Path(entry_dir, f'{model.__class__.__name__}.{pred_var}.json'))

            metrics[market_date] = cv.metrics.metrics
            pd.DataFrame(metrics).T.to_csv(pathlib.Path(dump_dir, f'metrics.{pred_var}.csv'))

    def reset(self):
        """
        Resets the factor and factor_value data.
        """
        self.factor.clear()
        MDS.clear()

        self.factor_value.clear()
        self.decoder.clear()
        for _ in self.cv.values():
            _.clear()

    def run(self):
        """
        Runs the factor validation process.
        """
        # self.initialize_factor()

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

        for market_data in replay:  # type: MarketData
            if not is_market_session(market_data.timestamp):
                continue

            MDS.on_market_data(market_data=market_data)

            timestamp_index = int(market_data.timestamp // self.sampling_interval) * self.sampling_interval

            if timestamp_index not in self.factor_value:
                entry_log = self.factor_value[timestamp_index] = {}
                self._collect_factor(entry_log=entry_log)
            else:
                entry_log = self.factor_value[timestamp_index]

            entry_log[f'{market_data.ticker}.market_price'] = market_data.market_price

    def bod(self, market_date: datetime.date, replay: ProgressiveReplay, **kwargs) -> None:
        LOGGER.info(f'Starting {market_date} bod process...')

        # Startup task 0: Update subscription
        self._update_index_weights(market_date=market_date)

        # Backtest specific action 1: Unzip data
        historical.unzip_batch(market_date=market_date, ticker_list=self.index_weights.keys())

        # Startup task 2: Update subscription and replay
        self._update_subscription(replay=replay)

        # Startup task 3: Update caches
        self.initialize_factor()

    def eod(self, market_date: datetime.date, replay: ProgressiveReplay, **kwargs) -> None:
        random.seed(42)
        LOGGER.info(f'Starting {market_date} eod process...')

        self.validation(market_date=market_date)

        self.dump_result(market_date=market_date)

        self.reset()


class FactorBatchValidation(FactorValidation):

    def __init__(self, **kwargs):
        """
        Initializes the FactorBatchValidation instance.

        Args:
            **kwargs: Additional parameters for configuration.
        """
        super().__init__(
            poly_degree=kwargs.pop('poly_degree', 2),
            **kwargs
        )

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
            'TradeFlow.Adaptive.Index.Imbalance',
            'TradeFlow.Adaptive.Index.Entropy',
            'TradeFlow.Adaptive.Index.Boosted',
            'TradeFlow.Adaptive.Index.Slope',
            # 'Aggressiveness.EMA.Index',
        ]
        self.factor: list[FactorMonitor] = []

        self.factor_pool = FACTOR_POOL
        self.factor_cache = FactorPoolDummyMonitor(factor_pool=self.factor_pool)

    def _collect_factor(self, entry_log: dict[str, float]):
        # only un-cached monitors is registered
        super()._collect_factor(entry_log=entry_log)

        if not self.override_cache:
            factors = collect_factor(monitors=self.factor_cache)
            entry_log.update(factors)

    def _plot_factors(self, factors: pd.DataFrame, precision=4):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Select relevant columns from factors
        selected_factors = factors[self.features]
        hover_data = factors[self.features].astype(np.float64)

        # Create subplot
        fig = make_subplots(
            rows=len(self.features) + 1,
            cols=1,
            shared_xaxes=False,
            subplot_titles=['Synthetic'] + self.features,
            row_heights=[3] + [1] * len(self.features)
        )

        candlestick_trace = self._candle_sticks(factor_value=factors)
        candlestick_trace['name'] = 'Synthetic'
        candlestick_trace['showlegend'] = True
        fig.add_trace(candlestick_trace, row=1, col=1)
        fig['layout'][f'yaxis']['title'] = 'Synthetic'

        # Add traces for each feature
        for i, feature in enumerate(self.features):
            trace = go.Scatter(
                x=candlestick_trace['x'],
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

    def initialize_factor(self, **kwargs) -> list[FactorMonitor]:
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
            TradeFlowAdaptiveIndexMonitor(
                sampling_interval=3 * 5,
                sample_size=20,
                baseline_window=100,
                weights=self.index_weights,
                aligned_interval=False
            ),
            # AggressivenessEMAMonitor(
            #     discount_interval=1,
            #     alpha=ALPHA_0001,
            #     weights=self.index_weights
            # )
        ]

        for _ in self.factor:
            MDS.add_monitor(_)

        MDS.add_monitor(self.synthetic)

        if not self.override_cache:
            MDS.add_monitor(self.factor_cache)

        return self.factor

    def initialize_cache(self, market_date: datetime.date, replay: ProgressiveReplay):
        if self.override_cache:
            return

        self.factor_pool.load(market_date=market_date, factor_dir=self.cache_dir)
        factor_existed = self.factor_pool.factor_names(market_date=market_date)

        for factor in self.factor:
            factor_names = factor.factor_names(subscription=list(self.subscription))

            if all([_ in factor_existed for _ in factor_names]):
                factor.enabled = False
                LOGGER.info(f'Factor {factor.name} found in the factor cache, and will be disabled.')

        # no replay task is needed, remove all tasks
        if all([not factor.enabled for factor in self.factor]):
            replay.replay_subscription.clear()
            self.subscription.clear()  # need to ensure the synchronization of the subscription
            LOGGER.info(f'{market_date} All factor is cached, skip this day.')
            self.factor_value.update(self.factor_pool.storage[market_date])

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

    def dump_result(self, market_date: datetime.date):
        dump_dir = f'{self.__class__.__name__}.{self.validation_id}'
        os.makedirs(dump_dir, exist_ok=True)

        factor_value = pd.DataFrame(self.factor_value).T

        entry_dir = pathlib.Path(dump_dir, f'{market_date:%Y-%m-%d}')
        os.makedirs(entry_dir, exist_ok=True)

        if len(self.factor) > 2:
            file_name = f'{self.__class__.__name__}'
        else:
            file_name = f'{"".join([f"[{factor.name}]" for factor in self.factor])}.validation'

        factor_value.to_csv(pathlib.Path(entry_dir, f'{file_name}.factors.csv'))

        fig = self._plot_factors(factors=factor_value)
        fig.write_html(pathlib.Path(entry_dir, f'{file_name}.factor.html'))

        for pred_var in self.pred_var:
            model = self.model[pred_var]
            cv = self.cv[pred_var]
            metrics = self.metrics[pred_var]

            if cv.x_val is not None:
                fig = self._plot_cv(pred_var=pred_var, factors=factor_value)
                fig.write_html(pathlib.Path(entry_dir, f'{file_name}.{pred_var}.pred.html'))

                model.dump(pathlib.Path(entry_dir, f'{file_name}.{pred_var}.model.json'))
                cv.metrics.to_html(pathlib.Path(entry_dir, f'{file_name}.{pred_var}.metrics.html'))

                metrics[market_date] = cv.metrics.metrics
                pd.DataFrame(metrics).T.to_csv(pathlib.Path(dump_dir, f'metrics.{pred_var}.csv'))

    def bod(self, market_date: datetime.date, replay: ProgressiveReplay, **kwargs) -> None:

        super().bod(market_date=market_date, replay=replay, **kwargs)

        self.initialize_cache(market_date=market_date, replay=replay)

    def eod(self, market_date: datetime.date, **kwargs) -> None:

        self.update_cache(market_date=market_date)

        super().eod(market_date=market_date, **kwargs)

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

        for _ in self.cv.values():
            _.clear()


class InterTemporalValidation(FactorBatchValidation):
    """
    model is trained prior to the beginning of the day, using multi-day data
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.training_days: int = kwargs.get('training_days', 5)
        self.factor_value_storage = deque(maxlen=self.training_days)

    def _out_sample_validation(self, x_train, y_train, x_val, y_val, factors: pd.DataFrame, cv: CrossValidation):
        x_axis = np.array([datetime.datetime.fromtimestamp(_, tz=TIME_ZONE) for _ in factors.index])

        valid_mask = np.all(np.isfinite(x_train), axis=1) & np.isfinite(y_train)
        x_train = x_train[valid_mask]
        y_train = y_train[valid_mask]

        valid_mask = np.all(np.isfinite(x_val), axis=1) & np.isfinite(y_val)
        x_val = x_val[valid_mask]
        y_val = y_val[valid_mask]
        x_axis = x_axis[valid_mask]

        # if isinstance(self.model, RidgeRegression):
        #     self.model.optimal_alpha(x=x_train, y=y_train)

        cv.validate(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
        cv.x_axis = x_axis

    def validation(self, market_date: datetime.date):
        self.factor_value_storage.append(self.factor_value.copy())
        LOGGER.info(f'{market_date} validation started with {len(self.factor_value_storage):,} days obs.')

        factor_value = pd.DataFrame(self.factor_value).T

        for pred_var in self.pred_var:
            cv = self.cv[pred_var]

            # Step 1: define training set
            x_list, y_list = [], []
            for factor_value in self.factor_value_storage:
                factor_value = pd.DataFrame(factor_value).T
                self.decoder.clear()

                _x = define_inputs(factor_value=factor_value, input_vars=self.features, poly_degree=self.poly_degree).to_numpy()
                _y = define_prediction(factor_value=factor_value, pred_var=pred_var, decoder=self.decoder, key=self.pred_target)
                _y = (_y - np.nanmedian(_y)).to_numpy()

                x_list.append(_x)
                y_list.append(_y)

            if len(x_list) <= 1:
                return

            x_train = np.concatenate(x_list[:-1])
            y_train = np.concatenate(y_list[:-1])
            x_val, y_val = x_list[-1], y_list[-1]

            # Step 2: define validation set
            self._out_sample_validation(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, factors=factor_value, cv=cv)


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
            # 'Skewness.PricePct.Index.Adaptive.Index',
            # 'Skewness.PricePct.Index.Adaptive.Slope',
            # 'Gini.PricePct.Index.Adaptive',
            # 'Coherence.Price.Adaptive.up',
            # 'Coherence.Price.Adaptive.down',
            # 'Coherence.Price.Adaptive.ratio',
            # 'Coherence.Volume.up',
            # 'Coherence.Volume.down',
            # 'Entropy.Price.Adaptive',
            # 'Entropy.Price',
            # 'Entropy.PricePct.Adaptive',
            # 'EMA.Divergence.Index.Adaptive.Index',
            # 'EMA.Divergence.Index.Adaptive.Diff',
            # 'EMA.Divergence.Index.Adaptive.Diff.EMA',
            'TradeFlow.Adaptive.Index.Imbalance',
            # 'TradeFlow.Adaptive.Index.MutualInfo',
            'TradeFlow.Adaptive.Index.Boosted',
            'TradeFlow.Adaptive.Index.Slope',
        ]

        self.cache_dir = kwargs.get('cache_dir', pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res', 'tmp_factor_cache'))

        # experimenting with the factors requires clearing caches regularly
        if os.path.isdir(self.cache_dir) and self.override_cache:
            shutil.rmtree(self.cache_dir)  # remove dir and all contains
            LOGGER.info(f'Factor cache {self.cache_dir} removed!')

    def initialize_factor(self, **kwargs) -> list[FactorMonitor]:
        """
        Initializes multiple factors for validation.

        Args:
            **kwargs: Additional parameters for factor configuration.

        Returns:
            list[MarketDataMonitor]: Initialized list of market data monitors.
        """
        self.factor = [
            # CoherenceAdaptiveMonitor(
            #     sampling_interval=15,
            #     sample_size=20,
            #     baseline_window=100,
            #     weights=self.index_weights,
            #     center_mode='median',
            #     aligned_interval=True
            # ),
            # TradeCoherenceMonitor(
            #     sampling_interval=15,
            #     sample_size=20,
            #     weights=self.index_weights
            # ),
            # EntropyMonitor(
            #     sampling_interval=15,
            #     sample_size=20,
            #     weights=self.index_weights
            # ),
            # EntropyAdaptiveMonitor(
            #     sampling_interval=15,
            #     sample_size=20,
            #     weights=self.index_weights
            # ),
            # EntropyAdaptiveMonitor(
            #     sampling_interval=15,
            #     sample_size=20,
            #     weights=self.index_weights,
            #     ignore_primary=False,
            #     pct_change=True,
            #     name='Monitor.Entropy.PricePct.Adaptive'
            # ),
            # GiniIndexAdaptiveMonitor(
            #     sampling_interval=3 * 5,
            #     sample_size=20,
            #     baseline_window=100,
            #     weights=self.index_weights
            # ),
            # SkewnessIndexAdaptiveMonitor(
            #     sampling_interval=3 * 5,
            #     sample_size=20,
            #     baseline_window=100,
            #     weights=self.index_weights,
            #     aligned_interval=False
            # ),
            # DivergenceIndexAdaptiveMonitor(
            #     weights=self.index_weights,
            #     sampling_interval=15,
            #     baseline_window=20,
            # ),
            TradeFlowAdaptiveIndexMonitor(
                sampling_interval=3 * 5,
                sample_size=20,
                baseline_window=100,
                weights=self.index_weights,
                aligned_interval=False
            ),
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

    # validator = FactorValidation(
    #     start_date=datetime.date(2023, 1, 1),
    #     end_date=datetime.date(2023, 2, 1)
    # )

    # validator = FactorBatchValidation(
    #     start_date=datetime.date(2023, 1, 1),
    #     end_date=datetime.date(2023, 2, 1)
    # )

    validator = InterTemporalValidation(
        start_date=datetime.date(2023, 1, 1),
        end_date=datetime.date(2023, 4, 1),
        training_days=5,
    )

    # validator = FactorValidatorExperiment(
    #     override_cache=True,
    #     start_date=datetime.date(2023, 1, 1),
    #     end_date=datetime.date(2023, 2, 1)
    # )

    validator.run()
    safe_exit()


if __name__ == '__main__':
    main()
