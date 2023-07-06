__package__ = 'Quark.Backtest'

import datetime
import os.path
import pathlib
import time
import uuid

import simulated_env
from . import LOGGER
from ..API import historical
from ..Base import GlobalStatics
from ..Calibration.linear import LinearCore
from ..Misc import helper
from ..Strategy.data_core import SyntheticIndexMonitor, VolatilityMonitor
from ..Strategy.decision_core import DummyDecisionCore
from ..Strategy.strategy import Strategy, StrategyStatus

INDEX_NAME = '000016.SH'
MARKET_DATE = START_DATE = datetime.date(2023, 1, 1)
END_DATE = datetime.date(2023, 6, 1)
TEST_ID = str(uuid.uuid4())
TEST_ID_SHORT = TEST_ID.split('-')[0]
IS_INITIALIZED = False
EPOCH_TS = 0.
CALENDAR = simulated_env.trade_calendar(start_date=START_DATE, end_date=END_DATE)

STRATEGY = Strategy(
    index_ticker=INDEX_NAME,
    index_weights=helper.load_dict(
        file_path=pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res', f'index_weights.{INDEX_NAME}.{MARKET_DATE:%Y%m%d}.json'),
        json_dict=simulated_env.query_index_weights(index_name=INDEX_NAME, market_date=MARKET_DATE)
    ),
    mode='sampling'
)


def init_cache_daily(tickers: list[str], look_back: int = 90):
    import pickle

    start_date = START_DATE - datetime.timedelta(days=look_back)
    end_date = END_DATE

    cache_path = pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res', f'data_cache.{INDEX_NAME}.{start_date:%Y%m%d}.{end_date:%Y%m%d}.pkl')

    if os.path.isfile(cache_path):
        with open(cache_path, 'rb') as f:
            simulated_env.DATA_CACHE.update(pickle.load(f))
        return

    for ticker in tickers:
        LOGGER.info(f'initializing cache for {ticker} from {start_date}, {end_date}')
        simulated_env.preload_daily_cache(ticker=ticker, start_date=start_date, end_date=end_date)

    with open(cache_path, 'wb') as f:
        pickle.dump(simulated_env.DATA_CACHE, f)


def bod(market_date: datetime.date, **kwargs):
    global MARKET_DATE
    global IS_INITIALIZED
    global EPOCH_TS

    if market_date not in CALENDAR:
        return

    MARKET_DATE = market_date
    dump_dir = pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, f'TestResult.{TEST_ID_SHORT}')

    # startup task 0: load index weights
    index_weights = simulated_env.query_index_weights(index_name=INDEX_NAME, market_date=MARKET_DATE)
    STRATEGY.index_weights = index_weights

    # backtest specific action 0: initializing cache
    if not IS_INITIALIZED:
        init_cache_daily(tickers=list(index_weights.keys()) + [INDEX_NAME])

    # startup task 1: update subscription
    subscription = set(index_weights.keys())

    # backtest specific action 1: unzip data
    historical.unzip_batch(market_date=market_date, ticker_list=subscription)
    STRATEGY.subscription.clear()

    if 'replay' in kwargs:
        replay = kwargs['replay']

        for _ in subscription:
            if _ not in STRATEGY.subscription:
                replay.add_subscription(ticker=_, dtype='TradeData')

        for _ in STRATEGY.subscription:
            if _ not in subscription:
                replay.remove_subscription(ticker=_, dtype='TradeData')

    STRATEGY.subscription.update(subscription)

    # startup task 2: update monitors
    # for backtest purpose, the monitors is only registered once
    if not IS_INITIALIZED:
        monitors = STRATEGY.register()
    else:
        monitors = STRATEGY.monitors
        STRATEGY.status = StrategyStatus.working

    # OPTIONAL: task 2.1: update baseline for Monitor.SyntheticIndex
    monitor: SyntheticIndexMonitor = monitors.get('Monitor.SyntheticIndex')
    if monitor:
        last_close_price = helper.load_dict(
            file_path=pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res', f'last_close.{MARKET_DATE:%Y%m%d}.json'),
            json_dict={_: simulated_env.query_daily(ticker=_, market_date=MARKET_DATE, key='preclose') for _ in index_weights}  # in production, delete this line
        )
        monitor.base_price.clear()
        monitor.base_price.update(last_close_price)
        monitor.synthetic_base_price = simulated_env.query_daily(ticker=INDEX_NAME, market_date=MARKET_DATE, key='preclose')

    # OPTIONAL: task 2.2: update baseline for Monitor.VolatilityMonitor
    monitor: VolatilityMonitor = monitors.get('Monitor.Volatility.Daily')
    if monitor:
        last_close_price = helper.load_dict(
            file_path=pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res', f'last_close.{MARKET_DATE:%Y%m%d}.json'),
            json_dict={_: simulated_env.query_daily(ticker=_, market_date=MARKET_DATE, key='preclose') for _ in index_weights}  # in production, delete this line
        )
        monitor.base_price.clear()
        monitor.base_price.update(last_close_price)
        monitor.synthetic_base_price = simulated_env.query_daily(ticker=INDEX_NAME, market_date=MARKET_DATE, key='preclose')
        daily_volatility = helper.load_dict(
            file_path=pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res', f'volatility.daily.{MARKET_DATE:%Y%m%d}.json'),
            json_dict={_: simulated_env.query_volatility_daily(ticker=_, market_date=MARKET_DATE, window=20) for _ in index_weights}  # in production, delete this line
        )
        monitor.daily_volatility.clear()
        monitor.daily_volatility.update(daily_volatility)
        monitor.index_volatility = simulated_env.query_volatility_daily(ticker=INDEX_NAME, market_date=MARKET_DATE, window=20)

    # OPTIONAL: task 2.3: update baseline for Monitor.SyntheticIndex
    monitor: SyntheticIndexMonitor = monitors.get('Monitor.Decoder.Index')
    if monitor:
        last_close_price = helper.load_dict(
            file_path=pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res', f'last_close.{MARKET_DATE:%Y%m%d}.json'),
            json_dict={_: simulated_env.query_daily(ticker=_, market_date=MARKET_DATE, key='preclose') for _ in index_weights}  # in production, delete this line
        )
        monitor.base_price.clear()
        monitor.base_price.update(last_close_price)
        monitor.synthetic_base_price = simulated_env.query_daily(ticker=INDEX_NAME, market_date=MARKET_DATE, key='preclose')

    # startup task 3: initialize decision core
    try:
        STRATEGY.decision_core = LinearCore(ticker=INDEX_NAME, decode_level=3)
        STRATEGY.decision_core.load(file_dir=dump_dir, file_pattern=r'decision_core\.(\d{4}-\d{2}-\d{2})\.json')
    except FileNotFoundError as _:
        STRATEGY.decision_core = DummyDecisionCore()  # in production mode, just throw the error and stop the program

    # backtest-specific codes
    if not IS_INITIALIZED:
        IS_INITIALIZED = True

    EPOCH_TS = time.time()


def eod(market_date: datetime.date = MARKET_DATE, **kwargs):
    if market_date not in CALENDAR:
        return

    dump_dir = pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, f'TestResult.{TEST_ID_SHORT}')

    os.makedirs(dump_dir, exist_ok=True)

    STRATEGY.position_tracker.clear()
    STRATEGY.strategy_metric.dump(dump_dir.joinpath(f'metric.{MARKET_DATE}.csv'))
    STRATEGY.decision_core.calibrate(metric=STRATEGY.strategy_metric)
    STRATEGY.decision_core.dump(dump_dir.joinpath(f'decision_core.{MARKET_DATE}.json'))
    STRATEGY.strategy_metric.clear()
    STRATEGY.decision_core.clear()
    LOGGER.info(f'Backtest epoch {market_date} complete! Time costs {time.time() - EPOCH_TS}')


STRATEGY.engine.add_handler(on_bod=bod)
STRATEGY.engine.add_handler(on_eod=eod)

if __name__ == '__main__':
    STRATEGY.engine.back_test(
        start_date=START_DATE,
        end_date=END_DATE,
        data_loader=historical.loader,
        mode='sampling'
    )
