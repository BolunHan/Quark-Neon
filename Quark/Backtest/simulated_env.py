import datetime
import functools
import traceback

import baostock as bs
import numpy as np
import pandas as pd
import exchange_calendars

BAO_STOCK = None
DATA_CACHE = {}

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
        if self.login_type == 'bs':
            self.bs_login()
        else:
            raise NotImplementedError(f'Invalid login type {self.login_type}')
        return function


class Cache(object):
    def __init__(self, topic: str, overwrite=False):
        self.topic = topic
        self.overwrite = overwrite

        if topic in DATA_CACHE:
            self.cache = DATA_CACHE[topic]
        else:
            self.cache = DATA_CACHE[topic] = {}

    def __call__(self, function: callable):
        if self.topic == 'Daily':
            return functools.partial(self._daily_wrapper, f=function)
        else:
            return function

    def _daily_wrapper(self, f, **kwargs):
        ticker = kwargs['ticker']
        market_date = kwargs['market_date']
        key = f'{ticker}.{market_date:%Y-%m-%d}'

        if key not in self.cache or self.overwrite:
            self.cache[key] = f(**kwargs)

        return self.cache[key]


def trade_calendar(start_date: datetime.date, end_date: datetime.date, market='SSE') -> list[datetime.date]:
    calendar = exchange_calendars.get_calendar(market)
    sessions = calendar.sessions_in_range(start_date, end_date)
    return [_.date() for _ in sessions]


def query_index_weights(index_name: str, market_date: datetime.date):
    df = pd.read_csv(fr"C:\Users\Bolun\Projects\Quark\Res\weights.{index_name.split('.')[0]}.csv")
    weights = {}

    for _, row in df.iterrows():
        weights[row['Symbol']] = float(row['Weights'])

    return weights


def query_daily(ticker: str, market_date: datetime.date, key: str = 'close') -> float:
    daily = _query_daily(ticker=ticker, market_date=market_date)

    if daily is None:
        return np.nan
    else:
        return float(daily[_DAILY_MAPPING[key]])


@Cache(topic='Daily')
@Login(login_type='bs')
def _query_daily(**kwargs) -> list[str] | None:
    ticker: str = kwargs.pop('ticker')
    market_date: datetime.date = kwargs.pop('market_date')

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


def preload_daily_cache(ticker: str, start_date: datetime.date, end_date: datetime.date):
    if 'Daily' in DATA_CACHE:
        cache = DATA_CACHE['Daily']
    else:
        cache = DATA_CACHE['Daily'] = {}

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
