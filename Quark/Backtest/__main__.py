__package__ = 'Quark.Backtest'

import datetime
import os
import pathlib
import pickle
import random
import sys
import time
import traceback

import pandas as pd
from AlgoEngine.Engine import ProgressiveReplay, SimMatch
from PyQuantKit import MarketData, TickData

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from . import LOGGER
from . import simulated_env
from ..API import historical, external
from ..Base import GlobalStatics
from ..DataLore.data_lore import LinearDataLore
from ..DecisionCore import DummyDecisionCore, MajorityDecisionCore as Core
from ..Factor import *
from ..Factor.factor_pool import FACTOR_POOL, FactorPoolDummyMonitor
from ..Profile import cn
from ..Strategy import Strategy, MDS

DUMMY_WEIGHT = False
cn.profile_cn_override()


class BackTest(object):
    def __init__(self, index_name: str, start_date: datetime.date, end_date: datetime.date, **kwargs):
        # statics
        self.index_name = index_name
        self.start_date = start_date
        self.end_date = end_date
        self.override_cache = kwargs.get('override_cache', False)
        self.use_dummy_core = kwargs.get('dummy_core', False)
        self.sampling_interval = kwargs.get('sampling_interval', 10.)
        self.res_dir = kwargs.get('res_dir', pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res'))
        self.test_id = kwargs.get('test_id', self._get_test_id())
        self.dump_dir = kwargs.get('dump_dir', pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, f'TestResult.{self.test_id}'))
        self.calendar = simulated_env.trade_calendar(start_date=self.start_date, end_date=self.end_date)
        self.factor_pool = FACTOR_POOL
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

        # local variable (over each iteration)
        self.market_date: datetime.date = self.calendar[0]
        self.index_weights: IndexWeight = IndexWeight(self.index_name, simulated_env.query(ticker=self.index_name, market_date=self.market_date, topic='index_weights'))
        self.factor: list[FactorMonitor] = []
        self.factor_cache: FactorPoolDummyMonitor | None = None
        self.synthetic: SyntheticIndexMonitor = SyntheticIndexMonitor(index_name=self.index_name, weights=self.index_weights)

        # flags
        self.epoch_ts = 0.

        # initialize strategy
        self.strategy: Strategy | None = self._get_strategy()
        self.engine = self.strategy.engine
        self.event_engine = self.engine.event_engine
        self.topic_set = self.engine.topic_set
        self.sim_match = {}
        self.fake_market_data: list[TickData] = []

    def _preload_cache(self, look_back: int = 90):
        start_date = self.start_date - datetime.timedelta(days=look_back)
        end_date = self.end_date

        cache_path = pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res', f'data_cache.{self.index_name}.{start_date:%Y%m%d}.{end_date:%Y%m%d}.pkl')

        if os.path.isfile(cache_path):
            external.load_cache(cache_file=cache_path)
            return

        for ticker in list(self.index_weights) + [self.index_name]:
            LOGGER.info(f'initializing cache for {ticker} from {start_date}, {end_date}')
            simulated_env.preload_daily_cache(ticker=ticker, start_date=start_date, end_date=end_date)

        simulated_env.dump_cache(cache_file=cache_path)

    def _update_index_weights(self, market_date: datetime.date):
        index_weights = simulated_env.query(ticker=self.index_name, market_date=market_date, topic='index_weights')

        if DUMMY_WEIGHT:
            LOGGER.warning('Using dummy index weights for faster debugging.')
            for _ in list(index_weights.keys())[5:]:
                index_weights.pop(_)

        self.index_weights.clear()
        self.index_weights.update(index_weights)
        self.index_weights.normalize()

        self.synthetic.weights = self.index_weights
        self.strategy.index_weights = self.index_weights

    def _update_subscription(self, replay: ProgressiveReplay):
        subscription = set(self.index_weights.keys())

        for _ in subscription:
            if _ not in self.strategy.subscription:
                replay.add_subscription(ticker=_, dtype='TradeData')

        for _ in self.strategy.subscription:
            if _ not in subscription:
                replay.remove_subscription(ticker=_, dtype='TradeData')

        self.strategy.subscription.clear()
        self.strategy.subscription.update(subscription)

        # add additional subscription for strategy: the index tick data (not available for replay api)
        self.strategy.subscription.add(self.index_name)

    @classmethod
    def _get_test_id(cls):
        test_id = 1

        while True:
            dump_dir = pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, f'TestResult.{test_id}')
            if os.path.isdir(dump_dir):
                test_id += 1
            else:
                break

        return test_id

    def _get_strategy(self):
        strategy = Strategy(
            index_ticker=self.index_name,
            index_weights=self.index_weights,
            mode='sampling'
        )

        strategy.subscription.update(self.index_weights.keys())
        strategy.engine.add_handler(on_bod=self.bod)
        strategy.engine.add_handler(on_eod=self.eod)
        return strategy

    def _sim_match_synthetic(self, market_data: MarketData, sampling_interval: float = 1.):
        timestamp = market_data.timestamp

        if self.index_name in self.sim_match:
            sim_match = self.sim_match[self.index_name]
        else:
            sim_match = self.sim_match[self.index_name] = SimMatch(ticker=self.index_name)
            sim_match.register(event_engine=self.event_engine, topic_set=self.topic_set)

        # the matching is done at external process
        if market_data.ticker == self.index_name:
            pass
        # avoid over-sampling to preserve resources
        elif self.fake_market_data and int(self.fake_market_data[-1].timestamp // sampling_interval) == int(timestamp // sampling_interval):
            pass
        # simulate the index data for match making and cache
        else:
            index_price = self.synthetic.index_price

            # simulate 2 index tick data
            fake_data = TickData(
                ticker=self.index_name,
                timestamp=timestamp,
                last_price=index_price
            )

            MDS.on_market_data(market_data=fake_data)

            self.engine.mds.on_market_data(market_data=fake_data)
            self.engine.position_tracker.on_market_data(market_data=fake_data)
            self.engine.__call__(market_data=fake_data)
            sim_match.__call__(market_data=fake_data)
            self.fake_market_data.append(fake_data)

    def initialize_factor(self) -> list[FactorMonitor]:
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
            )
        ]

        self.factor_cache = FactorPoolDummyMonitor(factor_pool=self.factor_pool)

        for _ in self.factor:
            MDS.add_monitor(_)
            self.strategy.monitors[_.name] = _

        MDS.add_monitor(self.synthetic)
        self.strategy.monitors[self.synthetic.name] = self.synthetic
        self.strategy.synthetic = self.synthetic

        if not self.override_cache:
            MDS.add_monitor(self.factor_cache)
            self.strategy.monitors[self.factor_cache.name] = self.factor_cache

        return self.factor

    def initialize_cache(self, market_date: datetime.date, replay: ProgressiveReplay):
        if self.override_cache:
            return

        # load fake data
        fake_data_path = self.res_dir.joinpath('Data', f'{self.index_name}.{market_date:%Y-%m-%d}.fake_data.pkl')
        if os.path.isfile(fake_data_path):
            LOGGER.info(f'Loading {market_date} cached {self.index_name} index data from {fake_data_path}')
            with open(fake_data_path, 'rb') as f:
                self.fake_market_data.extend(pickle.load(f))

        # load factor cache
        self.factor_pool.load(market_date=market_date, factor_dir=self.res_dir.joinpath('Factors'))
        factor_existed = self.factor_pool.factor_names(market_date=market_date)

        # mask cached monitors
        for factor in self.factor:
            factor_names = factor.factor_names(subscription=list(self.strategy.subscription))

            if all([_ in factor_existed for _ in factor_names]):
                factor.enabled = False
                LOGGER.info(f'Factor {factor.name} found in the factor cache, and will be disabled.')
            else:
                LOGGER.warning(f'Factor {factor.name} cache not hit! Recommend to use validation script first before back test.')

        # add fake data to replay
        if self.fake_market_data:
            if all([not factor.enabled for factor in self.factor]):
                replay.replay_subscription.clear()
                self.strategy.subscription.clear()  # need to ensure the synchronization of the subscription
                self.strategy.subscription.add(self.index_name)
                LOGGER.info(f'{market_date} All factor is cached, components replay tasks skipped.')

            # replay.add_subscription(ticker=self.index_name, dtype=TickData)
            replay.replay_task.extend(self.fake_market_data)
            LOGGER.info(f'{self.index_name} data cache found, added to the replay tasks.')

    def initialize_baseline(self, market_date: datetime.date):
        for monitor in self.factor + [self.synthetic]:
            if isinstance(monitor, SyntheticIndexMonitor):
                last_close_price = {_: simulated_env.query(ticker=_, market_date=market_date, topic='preclose') for _ in self.index_weights}
                monitor.base_price.update(last_close_price)
                monitor.synthetic_base_price = simulated_env.query(ticker=self.index_name, market_date=market_date, topic='preclose')
            elif isinstance(monitor, VolatilityMonitor):
                last_close_price = {_: simulated_env.query(ticker=_, market_date=market_date, topic='preclose') for _ in self.index_weights}
                monitor.base_price.update(last_close_price)
                monitor.synthetic_base_price = simulated_env.query(ticker=self.index_name, market_date=market_date, topic='preclose')
                daily_volatility = {_: simulated_env.query(ticker=_, market_date=market_date, topic='volatility') for _ in self.index_weights}
                monitor.daily_volatility.update(daily_volatility)
                monitor.index_volatility = simulated_env.query(ticker=self.index_name, market_date=market_date, topic='volatility')
            elif isinstance(monitor, IndexDecoderMonitor):
                last_close_price = {_: simulated_env.query(ticker=_, market_date=market_date, topic='preclose') for _ in self.index_weights}
                monitor.base_price.update(last_close_price)
                monitor.synthetic_base_price = simulated_env.query(ticker=self.index_name, market_date=market_date, topic='preclose')

    def initialize_decision_core(self, market_date: datetime.date):
        if self.use_dummy_core:
            return

        data_lore = LinearDataLore(
            ticker=self.index_name,
            alpha=0.05,
            trade_cost=0.0001,
            poly_degree=2,
            pred_length=15 * 60,
            bootstrap_samples=100,
            bootstrap_block_size=0.05
        )

        self.strategy.decision_core = Core(
            ticker=self.index_name,
            data_lore=data_lore
        )

        data_lore.inputs_var.clear()
        data_lore.inputs_var.extend(self.features)

        try:
            if os.path.isdir(self.dump_dir):
                self.strategy.decision_core = self.strategy.decision_core.load(
                    file_dir=self.dump_dir,
                    file_pattern=r'decision_core\.(\d{4}-\d{2}-\d{2})\.json'
                )
            else:
                LOGGER.error(f'{market_date} decision core can not be loaded, invalid directory {self.dump_dir}.')
        except FileNotFoundError as _:
            file_path = self.dump_dir.joinpath(r'decision_core\.(\d{4}-\d{2}-\d{2})\.json')
            LOGGER.error(f'{market_date} decision core can not be loaded, invalid file path {file_path}.')
        except Exception as _:
            self.strategy.decision_core = DummyDecisionCore()  # in production mode, just throw the error and stop the program
            LOGGER.error(f'{market_date} failed to load decision core! Fallback to dummy core!\ntraceback:\n{traceback.format_exc()}')

    def initialize_position_management(self):
        from AlgoEngine.Strategies import RISK_PROFILE

        RISK_PROFILE.set_rule(ticker=self.index_name, key='max_trade_long', value=20)
        RISK_PROFILE.set_rule(ticker=self.index_name, key='max_trade_short', value=20)
        # RISK_PROFILE.set_rule(ticker=self.index_name, key='max_exposure_long', value=100)
        # RISK_PROFILE.set_rule(ticker=self.index_name, key='max_exposure_short', value=100)

    def update_cache(self, market_date: datetime):
        # dump fake market data
        fake_data_path = self.res_dir.joinpath('Data', f'{self.index_name}.{market_date:%Y-%m-%d}.fake_data.pkl')
        os.makedirs(fake_data_path.parent, exist_ok=True)
        with open(fake_data_path, 'wb') as f:
            pickle.dump(self.fake_market_data, f)

        # dump factor pool
        if self.override_cache:
            LOGGER.info(f'Cache {market_date} overridden!')
            self.factor_pool.batch_update(factors=self.strategy.strategy_metric.factor_value)
            self.factor_pool.dump(factor_dir=self.res_dir.joinpath('Factors'))
        else:
            exclude_keys = self.factor_pool.factor_names(market_date=market_date)
            self.factor_pool.batch_update(factors=self.strategy.strategy_metric.factor_value, exclude_keys=exclude_keys)

            if all([name in exclude_keys for name in pd.DataFrame(self.strategy.strategy_metric.factor_value).T.columns]):
                pass
            else:
                self.factor_pool.dump(factor_dir=self.res_dir.joinpath('Factors'))
                LOGGER.info('Cache updated!')

    def update_decision_core(self, market_date: datetime.date):
        # in production mode, the market_date in kwargs is optional
        self.strategy.decision_core.calibrate(
            factor_value=self.strategy.strategy_metric.factor_value,
            market_date=market_date,
            trace_back=5
        )

        self.strategy.decision_core.validation(
            factor_value=self.strategy.strategy_metric.factor_value,
            candle_sticks=self.strategy.strategy_metric.candle_sticks
        )

    def dump_result(self, market_date: datetime.date):
        dump_dir = pathlib.Path(self.dump_dir, f'{market_date:%Y-%m-%d}')
        os.makedirs(dump_dir, exist_ok=True)

        # dump decision core
        self.strategy.decision_core.dump(self.dump_dir.joinpath(f'decision_core.{market_date}.json'))

        # dump factor values
        self.strategy.strategy_metric.dump(dump_dir.joinpath(f'metrics.{market_date}.csv'))

        # dump trade summaries
        self.strategy.strategy_metric.trade_summary.to_csv(dump_dir.joinpath(f'trade.summary.{market_date}.csv'))

        # dump backtest prediction
        self.strategy.strategy_metric.plot_prediction().write_html(dump_dir.joinpath(f'metrics.prediction.{market_date}.html'))

        # dump backtest trades
        self.strategy.strategy_metric.plot_trades().write_html(dump_dir.joinpath(f'metrics.trades.{market_date}.html'))
        self.strategy.strategy_metric.signal_trade_metrics.info.to_csv(dump_dir.joinpath(f'trade.signal.{market_date}.csv'))
        self.strategy.strategy_metric.target_trade_metrics.info.to_csv(dump_dir.joinpath(f'trade.target.{market_date}.csv'))
        self.strategy.strategy_metric.actual_trade_metrics.info.to_csv(dump_dir.joinpath(f'trade.actual.{market_date}.csv'))

        # dump in-sample validations
        for pred_var, cv in self.strategy.decision_core.data_lore.cv.items():
            fig = cv.plot()
            fig.write_html(dump_dir.joinpath(f'val.{pred_var}.in_sample.html'))

    def reset(self):
        self.factor.clear()
        self.strategy.mds.clear()
        self.strategy.clear()
        self.factor_pool.clear()
        self.factor_cache.clear()
        self.sim_match.clear()
        self.fake_market_data.clear()

    def run(self, **kwargs):
        self._preload_cache()

        replay = ProgressiveReplay(
            loader=historical.loader,
            tickers=list(self.index_weights),
            dtype=['TradeData'],
            start_date=self.start_date,
            end_date=self.end_date,
            calendar=self.calendar,
            bod=self.bod,
            eod=self.eod,
            tick_size=kwargs.get('progress_tick_size', 0.001),
        )

        start_ts = 0.
        self.event_engine.start()

        for market_data in replay:  # type: MarketData
            ticker = market_data.ticker

            if not start_ts:
                start_ts = time.time()

            if ticker not in self.sim_match:
                _ = self.sim_match[ticker] = SimMatch(ticker=ticker)
                _.register(event_engine=self.event_engine, topic_set=self.topic_set)

            if self.strategy.engine.multi_threading:
                self.engine.lock.acquire()
                self.event_engine.put(topic=self.topic_set.push(market_data=market_data), market_data=market_data)
            else:
                self.engine.mds.on_market_data(market_data=market_data)
                self.engine.position_tracker.on_market_data(market_data=market_data)
                self.engine.__call__(market_data=market_data)
                self.sim_match[ticker](market_data=market_data)

            self._sim_match_synthetic(market_data=market_data)

        LOGGER.info(f'All done! time_cost: {time.time() - start_ts:,.3}s')

    def bod(self, market_date: datetime.date, replay: ProgressiveReplay, **kwargs) -> None:
        if market_date not in self.calendar:
            return

        LOGGER.info(f'Starting {market_date} bod process...')
        self.market_date = market_date

        # Startup task 0: Update subscription
        self._update_index_weights(market_date=market_date)

        # Backtest specific Startup action 1: Unzip data
        historical.unzip_batch(market_date=market_date, ticker_list=self.index_weights.keys())

        # Startup task 2: Update subscription and replay
        self._update_subscription(replay=replay)

        # Startup task 3: Update caches
        self.initialize_factor()

        # Backtest specific Startup task 3.1: Update caches
        self.initialize_cache(market_date=market_date, replay=replay)

        # startup task 4: initialize baseline
        self.initialize_baseline(market_date=market_date)

        # startup task 5: override decision core
        self.initialize_decision_core(market_date=market_date)

        # startup task 6: register strategy
        self.strategy.register()

        # startup task 7: update risk profile
        self.initialize_position_management()

        # reset timer
        self.epoch_ts = time.time()

    def eod(self, market_date: datetime.date, **kwargs) -> None:
        if market_date not in self.calendar:
            return

        # EoD task -1: randomness cancellation patch
        random.seed(42)

        # EoD task 0: dump factor cache
        self.update_cache(market_date=market_date)

        # EoD task 1: calibrate decision core
        self.update_decision_core(market_date=market_date)

        # OPTIONAL EoD task 2: dump metrics, to validate factor cache
        self.dump_result(market_date=market_date)

        # OPTIONAL EoD task 3: clear and reset environment
        self.reset()

        LOGGER.info(f'Backtest epoch {market_date} complete! Time costs {time.time() - self.epoch_ts}')


def main():
    MDS.init_cn_override()

    tester = BackTest(
        index_name='000016.SH',
        start_date=datetime.date(2023, 1, 1),
        end_date=datetime.date(2023, 2, 1)
    )

    tester.run()


if __name__ == '__main__':
    main()
