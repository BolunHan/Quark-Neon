import datetime
import functools
import os
import pathlib
import pickle

import exchange_calendars
import numpy as np

from ..Base import GlobalStatics

CACHE = {}

_DAILY_FIELDS = "date,code,open,high,low,close,preclose,volume,amount"
_DAILY_MAPPING = {_DAILY_FIELDS.split(',')[_]: _ for _ in range(len(_DAILY_FIELDS.split(',')))}


class CacheMissed(ValueError):
    pass


class ExternalCache(object):
    def __init__(self, topic: str):
        self.topic = topic

    def __call__(self, function: callable):
        if self.topic == 'Daily':
            return functools.partial(self._parse_daily_cache, f=function)
        elif self.topic == 'IndexWeights':
            return functools.partial(self._parse_index_weights_cache, f=function)
        elif self.topic == 'Volatility':
            return functools.partial(self._parse_volatility_cache, f=function)
        else:
            return function

    def cache_wrapper(self, f, cache_key: str, **kwargs):
        if cache_key not in self.cache:
            raise CacheMissed(f'Cache not found for {self.topic} {cache_key}')

        return self.cache[cache_key]

    def _parse_daily_cache(self, f, **kwargs):
        ticker = kwargs['ticker']
        market_date = kwargs['market_date']
        key = f'{ticker}.{market_date:%Y-%m-%d}'
        return self.cache_wrapper(f=f, cache_key=key, **kwargs)

    def _parse_index_weights_cache(self, f, **kwargs):
        index_name = kwargs['index_name']
        market_date = kwargs['market_date']
        key = f'{index_name}.{market_date:%Y-%m-%d}'
        return self.cache_wrapper(f=f, cache_key=key, **kwargs)

    def _parse_volatility_cache(self, f, **kwargs):
        ticker = kwargs['ticker']
        market_date = kwargs['market_date']
        window = int(kwargs['window'])
        key = f'{ticker}.{market_date:%Y-%m-%d}.{window:,}'
        return self.cache_wrapper(f=f, cache_key=key, **kwargs)

    @property
    def cache(self):
        if self.topic in CACHE:
            cache = CACHE[self.topic]
        else:
            cache = CACHE[self.topic] = {}

        return cache


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


@ExternalCache(topic='Daily')
def _query_daily(**kwargs) -> list[str] | None:
    raise CacheMissed(f'No cache for {kwargs}')


@ExternalCache(topic='IndexWeights')
def _query_index_weights(**kwargs):
    raise CacheMissed(f'No cache for {kwargs}')


@ExternalCache(topic='Volatility')
def _query_volatility_daily(**kwargs) -> float:
    raise CacheMissed(f'No cache for {kwargs}')


def load_cache(market_date: datetime.date = None, cache_file: str | pathlib.Path = None, cache_dir: str | pathlib.Path = None):

    if cache_dir is None:
        cache_dir = pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res')

    if cache_file is None and market_date is None:
        cache_path = pathlib.Path(cache_dir, f'data_cache.pkl')
    elif cache_file is None:
        cache_path = pathlib.Path(cache_dir, f'data_cache.{market_date:%Y%m%d}.pkl')
    else:
        # cache_path = pathlib.Path(cache_dir, pathlib.Path(cache_file).name)
        cache_path = cache_file

    if os.path.isfile(cache_path):
        with open(cache_path, 'rb') as f:
            CACHE.update(pickle.load(f))
