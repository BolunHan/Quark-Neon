import datetime
from .Base import GlobalStatics
import pathlib
from .Strategy.strategy import Strategy, StrategyStatus
from .Misc import helper
from .Strategy.data_core import *
from .Calibration.linear import LinearCore

MARKET_DATE = datetime.date.today()
INDEX_NAME = '000016.SH'
STRATEGY = Strategy(
    index_ticker=INDEX_NAME,
    index_weights=helper.load_dict(
        file_path=pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res', f'index_weights.{INDEX_NAME}.{MARKET_DATE:%Y%m%d}.json'),
    ),
    mode='production'
)


# startup task 1: update subscription

def init_strategy(index_weights: dict[str, float]):
    subscription = set(index_weights.keys())
    STRATEGY.subscription.update(subscription)
    STRATEGY.status = StrategyStatus.working

    # startup task 3: initialize decision core
    try:
        STRATEGY.decision_core = LinearCore(ticker=INDEX_NAME, decode_level=3)
        STRATEGY.decision_core.load(file_path=pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res', f'decision_core.{MARKET_DATE:%Y%m%d}.json'))
    except Exception as _:
        LOGGER.warning('decision core not loaded, using dummy core!')

def init_monitors():
    monitors = STRATEGY.register()

    # OPTIONAL: task 2.1: update baseline for Monitor.SyntheticIndex
    monitor: SyntheticIndexMonitor = monitors.get('Monitor.SyntheticIndex')
    if monitor:
        last_close_price = helper.load_dict(file_path=pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res', f'last_close.{MARKET_DATE:%Y%m%d}.json'))
        monitor.base_price.clear()
        monitor.synthetic_base_price = last_close_price.pop(INDEX_NAME)
        monitor.base_price.update(last_close_price)

    # OPTIONAL: task 2.2: update baseline for Monitor.VolatilityMonitor
    monitor: VolatilityMonitor = monitors.get('Monitor.Volatility.Daily')
    if monitor:
        last_close_price = helper.load_dict(file_path=pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res', f'last_close.{MARKET_DATE:%Y%m%d}.json'))
        monitor.base_price.clear()
        monitor.synthetic_base_price = last_close_price.pop(INDEX_NAME)
        monitor.base_price.update(last_close_price)

        daily_volatility = helper.load_dict(file_path=pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res', f'volatility.daily.{MARKET_DATE:%Y%m%d}.json'))
        monitor.daily_volatility.clear()
        monitor.index_volatility = daily_volatility.pop(INDEX_NAME)
        monitor.daily_volatility.update(daily_volatility)

    # OPTIONAL: task 2.3: update baseline for Monitor.SyntheticIndex
    monitor: SyntheticIndexMonitor = monitors.get('Monitor.Decoder.Index')
    if monitor:
        last_close_price = helper.load_dict(file_path=pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res', f'last_close.{MARKET_DATE:%Y%m%d}.json'))
        monitor.base_price.clear()
        monitor.synthetic_base_price = last_close_price.pop(INDEX_NAME)
        monitor.base_price.update(last_close_price)
