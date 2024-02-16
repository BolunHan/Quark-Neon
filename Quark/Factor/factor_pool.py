"""
calculate and save factors to accelerate backtesting
"""
import csv
import datetime
import os.path
import pathlib
import re
from typing import Iterable

from PyQuantKit import MarketData

from . import LOGGER, FactorMonitor
from ..Base import GlobalStatics, CONFIG

TIME_ZONE = GlobalStatics.TIME_ZONE
LOGGER = LOGGER.getChild('FactorPool')


class FactorPool(object):

    def __init__(self, **kwargs):
        self.factor_dir = kwargs.get('factor_dir', pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res', 'Factors'))
        self.log_interval = kwargs.get('log_interval', CONFIG.Statistics.FACTOR_SAMPLING_INTERVAL)

        self.storage: dict[datetime.date, dict[float, dict[str, float]]] = {}  # market_date -> timestamp -> entry_key -> entry_value

    def update(self, timestamp: float, key: str, value: float | int | str):
        """
        update a single value of factor pool.
        """
        market_time = datetime.datetime.fromtimestamp(timestamp, tz=TIME_ZONE)
        market_date = market_time.date()

        storage = self.storage.get(market_date, {})

        timestamp = self.locate_timestamp(timestamp, key_range=storage.keys(), step=self.log_interval)
        storage[timestamp][key] = value

        self.storage[market_date] = storage

    def batch_update(self, factors: dict[float, dict[str, float]], include_keys: list[str] = None, exclude_keys: list[str] = None):
        """
        batch update assume that the factors is from the same date.
        update the logs of factor pool
        then dump the logs
        """

        market_date = None
        storage = None

        for timestamp in factors:
            timestamp = (timestamp // self.log_interval) * self.log_interval

            if market_date is None:
                market_time = datetime.datetime.fromtimestamp(timestamp, tz=TIME_ZONE)
                market_date = market_time.date()

                if market_date not in self.storage:
                    self.load(market_date=market_date)

                self.storage[market_date] = storage = self.storage.get(market_date, {})

            factor_log = factors[timestamp]
            selected_factor = storage.get(timestamp, {})

            for entry_name in factor_log:
                if include_keys is not None and entry_name not in include_keys:
                    continue

                if exclude_keys is not None and entry_name in exclude_keys:
                    continue

                selected_factor[entry_name] = factor_log[entry_name]

            if selected_factor:
                storage[timestamp] = selected_factor

    @classmethod
    def locate_timestamp(cls, timestamp: float, key_range: Iterable[float] = None, step: float = None) -> float:
        if key_range is None:
            return (timestamp // step) * step

        for _ in key_range:
            if _ < timestamp:
                continue

            return _

        return (timestamp // step) * step

    @classmethod
    def locate_cache_index(cls, timestamp: float, key_range: list[float] = None, start_idx: int = 0) -> int:
        i = start_idx

        for i in range(start_idx, len(key_range)):
            if key_range[i] <= timestamp:
                continue
            break
        return i

    def dump(self, factor_dir: str | pathlib.Path = None):

        if factor_dir is None:
            factor_dir = self.factor_dir

        if not os.path.isdir(factor_dir):
            os.makedirs(factor_dir, exist_ok=True)

        for market_date, storage in self.storage.items():
            file_name = pathlib.Path(factor_dir, f'{market_date:%Y%m%d}.factor.csv')

            column_names = set()
            for factor_dict in storage.values():
                column_names.update(factor_dict.keys())
            column_names = sorted(column_names)

            # Write the factors to the CSV file
            with open(file_name, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Write the header row
                writer.writerow(['Timestamp'] + column_names)

                # Write the data rows
                for timestamp, factor_dict in storage.items():
                    row = [timestamp] + [factor_dict.get(column, '') for column in column_names]
                    writer.writerow(row)

    def load(self, market_date: datetime.date, factor_dir: str | pathlib.Path = None) -> dict[float, dict[str, float]]:

        if not isinstance(market_date, datetime.date):
            raise TypeError(f'Expect datetime.date, got {market_date}')

        if factor_dir is None:
            factor_dir = self.factor_dir

        file_name = pathlib.Path(factor_dir, f'{market_date:%Y%m%d}.factor.csv')
        storage = dict()

        if not os.path.isfile(file_name):
            self.storage[market_date] = storage
            return storage

        with open(file_name, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            # Read the header row to extract the column names
            header = next(reader)
            column_names = header[1:]  # Exclude the 'Timestamp' column

            # Iterate over the data rows
            for row in reader:
                timestamp = float(row[0])
                factor_dict = {}

                # Extract the factor values for each column
                for column, value in zip(column_names, row[1:]):
                    if value:
                        factor_dict[column] = float(value)

                # Store the factor dictionary in the loaded_factors dictionary
                storage[timestamp] = factor_dict

        self.storage[market_date] = storage
        return storage

    def locate_caches(self, market_date: datetime.date, size: int = 5, pattern: str = r"\d{8}\.factor\.csv", exclude_current: bool = False) -> list[pathlib.Path]:
        closest_files = []

        # Get all files in the directory
        files = [f for f in os.listdir(self.factor_dir) if re.match(pattern, f)]

        # Sort the files in increasing order
        files.sort()

        # Loop through the file names and compare with the market_date
        for file_name in files:
            file_date = datetime.datetime.strptime(file_name[:8], "%Y%m%d").date()

            if market_date is not None and (file_date > market_date or (exclude_current and file_date == market_date)):
                break

            closest_files.append(file_name)

        result = [pathlib.Path(self.factor_dir, _) for _ in closest_files[-size:]]

        return result

    def clear(self):
        self.storage.clear()

    def factor_names(self, market_date: datetime.date) -> set[str]:
        column_names = set()

        if market_date not in self.storage:
            storage = self.load(market_date=market_date)
        else:
            storage = self.storage[market_date]

        for factor_dict in storage.values():
            column_names.update(factor_dict.keys())

        column_names = set(sorted(column_names))
        return column_names


class FactorPoolDummyMonitor(FactorMonitor):
    """
    query factor value from local storage, by given timestamp

    Note: the factor value may have some missing values
    """

    def __init__(self, factor_pool: FactorPool = None):
        super().__init__(name='Monitor.FactorPool.Dummy')

        self.factor_pool = FACTOR_POOL if factor_pool is None else factor_pool
        self._is_ready = True
        self.timestamp = 0.
        self.cache_index = 0

    def __call__(self, market_data: MarketData, **kwargs):
        timestamp = market_data.timestamp

        if timestamp < self.timestamp:
            self.cache_index = 0
            LOGGER.warning(f'Factor pool cache requires market_data replayed in strict order for optimal performance. previous ts {self.timestamp}, new ts {timestamp}. cache idx is reset.')

        self.timestamp = market_data.timestamp

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        raise NotImplementedError('Factor Cache monitor should not be serialized in any cases.')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict):
        raise NotImplementedError('Factor Cache monitor should not be serialized in any cases.')

    def clear(self) -> None:
        self.cache_index = 0
        self.timestamp = 0.

        if self.factor_pool is not None:
            self.factor_pool.clear()

    def factor_names(self, subscription: list[str]) -> list[str]:
        market_date = datetime.datetime.fromtimestamp(self.timestamp, tz=TIME_ZONE).date()
        # factor_storage = self.factor_pool.storage.get(market_date)
        return list(self.factor_pool.factor_names(market_date=market_date))

    @property
    def value(self) -> dict[str, float]:
        market_date = datetime.datetime.fromtimestamp(self.timestamp, tz=TIME_ZONE).date()
        factor_storage = self.factor_pool.storage.get(market_date)

        if factor_storage is None:
            factor_storage = self.factor_pool.load(market_date=market_date)

        # key = self.factor_pool.locate_timestamp(
        #     timestamp=self.timestamp,
        #     key_range=list(factor_storage.keys()),
        #     step=self.factor_pool.log_interval
        # )

        key_range = list(factor_storage.keys())

        idx = self.factor_pool.locate_cache_index(timestamp=self.timestamp, key_range=key_range, start_idx=self.cache_index)

        if idx < len(key_range):
            key = key_range[idx]
        else:
            key = (self.timestamp // self.factor_pool.log_interval) * self.factor_pool.log_interval

        self.cache_index = idx

        value = factor_storage.get(key, {})
        return value

    @property
    def is_ready(self) -> bool:
        return self._is_ready


FACTOR_POOL = FactorPool()
__all__ = [FACTOR_POOL, FactorPool]
