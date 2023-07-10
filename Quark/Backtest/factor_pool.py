"""
calculate and save factors to accelerate backtesting
"""
import csv
import datetime
import os.path
import pathlib
from typing import Iterable

from ..Base import GlobalStatics


class FactorPool(object):
    def __init__(self, **kwargs):
        self.factor_dir = kwargs.get('factor_dir', pathlib.Path(GlobalStatics.WORKING_DIRECTORY, 'Res', 'Factors'))
        self.log_interval = kwargs.get('log_interval', 5)

        self.storage: dict[datetime.date, dict[float, dict[str, float]]] = {}  # market_date -> timestamp -> entry_key -> entry_value

    def update(self, timestamp: float, key: str, value: float | int | str):
        """
        update a single value of factor pool.
        """
        market_time = datetime.datetime.fromtimestamp(timestamp, tz=GlobalStatics.TIME_ZONE)
        market_date = market_time.date()

        storage = self.storage.get(market_date, {})

        timestamp = self.locate_timestamp(timestamp, key_range=storage.keys(), step=self.log_interval)
        storage[timestamp][key] = value

        self.storage[market_date] = storage

    def batch_update(self, factors: dict[float, dict[str, float]]):
        """
        batch update assume that the factors is from the same date.
        update the logs of factor pool
        then dump the logs
        """

        market_date = None
        storage = None

        for timestamp in factors:
            timestamp = timestamp // self.log_interval * self.log_interval

            if market_date is None:
                market_time = datetime.datetime.fromtimestamp(timestamp, tz=GlobalStatics.TIME_ZONE)
                market_date = market_time.date()

                if market_date not in self.storage:
                    self._load(factor_dir=self.factor_dir, market_date=market_date)

                storage = self.storage.get(market_date)

            storage[timestamp] = factors[timestamp]

        self.storage.pop(market_date)
        self.dump(factor_dir=self.factor_dir)

    @classmethod
    def locate_timestamp(cls, timestamp: float, key_range: Iterable[float] = None, step: float = None) -> float:
        if key_range is None:
            return timestamp // step * step

        for _ in key_range:
            if _ < timestamp:
                continue

            return _

        return timestamp // step * step

    def dump(self, factor_dir: str | pathlib.Path):

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

    def _load(self, factor_dir: str | pathlib.Path, market_date: datetime.date) -> dict[float, dict[str, float]]:
        file_name = pathlib.Path(factor_dir, f'{market_date:%Y%m%d}.factor.csv')
        storage = dict()

        if not os.path.isfile(file_name):
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

    def factor_names(self, market_date: datetime.date) -> set[str]:
        column_names = set()

        if market_date not in self.storage:
            storage = self._load(factor_dir=self.factor_dir, market_date=market_date)
        else:
            storage = self.storage[market_date]

        for factor_dict in storage.values():
            column_names.update(factor_dict.keys())

        column_names = set(sorted(column_names))
        return column_names


FACTOR_POOL = FactorPool()
__all__ = [FACTOR_POOL, FactorPool]
