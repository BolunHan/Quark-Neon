import datetime
import pathlib
import pickle
import traceback

import baostock as bs
import exchange_calendars
import numpy as np
import pandas as pd

from ..API.external import ExternalCache, CACHE
from ..Base import GlobalStatics

BAO_STOCK = None

_DAILY_FIELDS = "date,code,open,high,low,close,preclose,volume,amount"
_DAILY_MAPPING = {_DAILY_FIELDS.split(',')[_]: _ for _ in range(len(_DAILY_FIELDS.split(',')))}


class Login(object):
    def __init__(self, login_type: str = 'bs', **kwargs):
        self.login_type = login_type
        self.parameter = kwargs

    def bs_login(self):
        global BAO_STOCK

        try:
            if BAO_STOCK is None:
                BAO_STOCK = bs.login(**self.parameter)
                if BAO_STOCK.error_code != '0':
                    raise PermissionError('Login Error error_code: {}; error_msg: {}'.format(BAO_STOCK.error_code, BAO_STOCK.error_msg))
        except Exception as _:
            raise ValueError(f'BaoStock login failed, {traceback.format_exc()}')

    def __call__(self, function: callable):

        def wrapper(*args, **kwargs):
            if self.login_type == 'bs':
                self.bs_login()
            else:
                raise NotImplementedError(f'Invalid login type {self.login_type}')

            return function(*args, **kwargs)

        return wrapper


class SimCache(ExternalCache):
    def __init__(self, topic: str, overwrite=False):
        super().__init__(topic=topic)
        self.overwrite = overwrite

    def cache_wrapper(self, f, cache_key: str, **kwargs):
        if cache_key not in self.cache or self.overwrite:
            _ = f(**kwargs)
            self.cache[cache_key] = _
            return _

        return self.cache[cache_key]


def trade_calendar(start_date: datetime.date, end_date: datetime.date, market='SSE') -> list[datetime.date]:
    calendar = exchange_calendars.get_calendar(market)
    sessions = calendar.sessions_in_range(start_date, end_date)
    return sorted([_.date() for _ in sessions])


def query(ticker: str, market_date: datetime.date, topic: str) -> float | dict[str, float]:
    if topic in _DAILY_MAPPING:
        daily = _query_daily(ticker=ticker, market_date=market_date)

        if daily is None:
            return np.nan
        else:
            return float(daily[_DAILY_MAPPING[topic]])
    elif topic == 'index_weights':
        return _query_index_weights(index_name=ticker, market_date=market_date)
    elif topic == 'volatility':
        return _query_volatility_daily(ticker=ticker, market_date=market_date, window=20)


@SimCache(topic='Daily')
@Login(login_type='bs')
def _query_daily(**kwargs) -> list[str] | None:
    ticker: str = kwargs.get('ticker')
    market_date: datetime.date = kwargs.get('market_date')

    symbol, market = ticker.split('.')
    rs = bs.query_history_k_data_plus(
        code=f'{market.lower()}.{symbol}',
        fields="date,code,open,high,low,close,preclose,volume,amount",
        start_date=f'{market_date:%Y-%m-%d}',
        end_date=f'{market_date:%Y-%m-%d}',
        frequency="d",
        adjustflag="3"
    )

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())

    if not data_list:
        return None
    else:
        return data_list[0]


@SimCache(topic='IndexWeights')
def _query_index_weights(**kwargs):
    index_name: str = kwargs.get('index_name')
    market_date: datetime.date = kwargs.get('market_date')

    df = pd.read_csv(pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res', f"index_weights.{index_name.split('.')[0]}.csv"))
    weights = {}

    for _, row in df.iterrows():
        weights[row['ticker']] = float(row['weight'])

    return weights


@SimCache(topic='Volatility')
def _query_volatility_daily(**kwargs) -> float:
    ticker: str = kwargs.get('ticker')
    market_date: datetime.date = kwargs.get('market_date')
    window: int = kwargs.get('window', 20)

    max_trade_day_before = int(np.ceil(window / 5) * 7 + 7)
    start_date = market_date - datetime.timedelta(days=max_trade_day_before)
    end_date = market_date

    calendar = trade_calendar(start_date=start_date, end_date=end_date)[-window:]

    pct_change_list = []

    for _ in calendar:
        pre_close = query(ticker=ticker, market_date=_, topic='preclose')
        close = query(ticker=ticker, market_date=_, topic='close')
        pct_change = close / pre_close - 1
        pct_change_list.append(pct_change)

    volatility = float(np.std(pct_change_list))
    return volatility


@Login(login_type='bs')
def preload_daily_cache(ticker: str, start_date: datetime.date, end_date: datetime.date):
    if 'Daily' in CACHE:
        cache = CACHE['Daily']
    else:
        cache = CACHE['Daily'] = {}

    symbol, market = ticker.split('.')
    rs = bs.query_history_k_data_plus(
        code=f'{market.lower()}.{symbol}',
        fields=_DAILY_FIELDS,
        start_date=f'{start_date:%Y-%m-%d}',
        end_date=f'{end_date:%Y-%m-%d}',
        frequency="d",
        adjustflag="3"
    )

    while (rs.error_code == '0') & rs.next():
        _ = rs.get_row_data()
        key = f'{ticker}.{_[0]}'
        cache[key] = _


def dump_cache(market_date: datetime.date = None, cache_file: str | pathlib.Path = None, cache_dir: str | pathlib.Path = None):
    if cache_dir is None:
        cache_dir = pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res')

    if cache_file is None and market_date is None:
        cache_path = pathlib.Path(cache_dir, f'data_cache.pkl')
    elif cache_file is None:
        cache_path = pathlib.Path(cache_dir, f'data_cache.{market_date:%Y%m%d}.pkl')
    else:
        # cache_path = pathlib.Path(cache_dir, pathlib.Path(cache_file).name)
        cache_path = cache_file

    with open(cache_path, 'wb') as f:
        pickle.dump(CACHE, f)
