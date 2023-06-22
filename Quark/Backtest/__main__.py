__package__ = 'Quark.Backtest'

import datetime
import os.path
import pathlib
import uuid

import simulated_env
from ..API import historical
from ..Base import GlobalStatics
from ..Misc import helper
from ..Strategy.data_core import SyntheticIndexMonitor
from ..Strategy.strategy import Strategy
from . import LOGGER

INDEX_NAME = '000016.SH'
MARKET_DATE = START_DATE = datetime.date(2023, 1, 1)
END_DATE = datetime.date(2023, 6, 1)
TEST_ID = str(uuid.uuid4())
TEST_ID_SHORT = TEST_ID.split('-')[0]
IS_INITIALIZED = False
CALENDAR = simulated_env.trade_calendar(start_date=START_DATE, end_date=END_DATE)

STRATEGY = Strategy(
    index_ticker=INDEX_NAME,
    index_weights=helper.load_dict(
        file_path=pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, f'index_weights.{INDEX_NAME}.{MARKET_DATE:%Y%m%d}.json'),
        json_dict=simulated_env.query_index_weights(index_name=INDEX_NAME, market_date=MARKET_DATE)
    )
)


def init_cache(tickers: list[str]):
    import pickle
    cache_path = pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, f'data_cache.{INDEX_NAME}.{START_DATE:%Y%m%d}.{END_DATE:%Y%m%d}.pkl')

    if os.path.isfile(cache_path):
        with open(cache_path, 'rb') as f:
            simulated_env.DATA_CACHE.update(pickle.load(f))
        return

    for ticker in tickers:
        LOGGER.info(f'initializing cache for {ticker} from {START_DATE}, {END_DATE}')
        simulated_env.preload_daily_cache(ticker=ticker, start_date=START_DATE, end_date=END_DATE)

    with open(cache_path, 'wb') as f:
        pickle.dump(simulated_env.DATA_CACHE, f)


def bod(market_date: datetime.date, **kwargs):
    global MARKET_DATE
    global IS_INITIALIZED

    if market_date not in CALENDAR:
        return

    MARKET_DATE = market_date

    # startup task 0: load index weights
    index_weights = simulated_env.query_index_weights(index_name=INDEX_NAME, market_date=MARKET_DATE)
    STRATEGY.index_weights = index_weights

    # backtest specific action 0: initializing cache
    if not IS_INITIALIZED:
        init_cache(tickers=list(index_weights.keys()) + [INDEX_NAME])

    # startup task 1: update subscription
    subscription = set(index_weights.keys())

    # backtest specific action 1: unzip data
    historical.unzip_batch(market_date=market_date, ticker_list=subscription)

    if 'replay' in kwargs:
        replay = kwargs['replay']

        for _ in subscription:
            if _ not in STRATEGY.subscription:
                replay.add_subscription(ticker=_, dtype='TradeData')

        for _ in STRATEGY.subscription:
            if _ not in subscription:
                replay.remove_subscription(ticker=_, dtype='TradeData')

    STRATEGY.subscription = subscription

    # startup task 2: update monitors
    # for backtest purpose, the monitors is only registered once
    if not IS_INITIALIZED:
        monitors = STRATEGY.register()
    else:
        monitors = STRATEGY.monitors

    # OPTIONAL: task 2.1: update baseline for Monitor.SyntheticIndex
    monitor: SyntheticIndexMonitor = monitors['Monitor.SyntheticIndex']
    last_close_price = helper.load_dict(
        file_path=pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, f'last_close.{MARKET_DATE:%Y%m%d}.json'),
        json_dict={_: simulated_env.query_daily(ticker=_, market_date=MARKET_DATE, key='preclose') for _ in index_weights}  # in production, delete this line
    )
    monitor.base_price.clear()
    monitor.base_price.update(last_close_price)
    monitor.index_base_price = simulated_env.query_daily(ticker=INDEX_NAME, market_date=MARKET_DATE, key='preclose')

    # backtest-specific codes
    if not IS_INITIALIZED:
        IS_INITIALIZED = True


def eod(market_date: datetime.date = MARKET_DATE, **kwargs):
    if market_date not in CALENDAR:
        return

    STRATEGY.position_tracker.clear()
    STRATEGY.strategy_metric.dump(pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, f'TestResult.{TEST_ID}', f'metric.{MARKET_DATE}.csv'))
    STRATEGY.strategy_metric.clear()


STRATEGY.engine.add_handler(on_bod=bod)
STRATEGY.engine.add_handler(on_eod=eod)

if __name__ == '__main__':
    STRATEGY.engine.back_test(
        start_date=START_DATE,
        end_date=END_DATE,
        data_loader=historical.loader
    )
