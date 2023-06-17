import csv
import datetime
import os.path
import pathlib

from PyQuantKit import TradeData

from . import LOGGER

ARCHIVE_DIR = r'C:\Users\Bolun\Downloads\逐步委托数据'
DATA_DIR = r'C:\Users\Bolun\Documents\TradeData'


def unzip(market_date: datetime.date, ticker: str):
    import py7zr

    archive_path = pathlib.Path(ARCHIVE_DIR, f'{market_date:%Y%m}', f'{market_date:%Y-%m-%d}.7z')
    destination_path = pathlib.Path(DATA_DIR, f'{market_date:%Y-%m-%d}')
    file_to_extract = f'{ticker.split(".")[0]}.csv'

    if not os.path.isfile(archive_path):
        raise FileNotFoundError(f'{archive_path} not found!')

    os.makedirs(destination_path, exist_ok=True)

    LOGGER.info(f'Unzipping {file_to_extract} from {archive_path} to {destination_path}...')

    with py7zr.SevenZipFile(archive_path, mode='r') as archive:
        archive.extract(targets=[file_to_extract], path=destination_path)

    return 0


def load_trade_data(market_date: datetime.date, ticker: str) -> list[TradeData]:
    trade_data_list = []

    file_path = pathlib.Path(DATA_DIR, f'{market_date:%Y-%m-%d}', f'{ticker.split(".")[0]}.csv')

    if not os.path.isfile(file_path):
        try:
            unzip(market_date=market_date, ticker=ticker)
        except FileNotFoundError as _:
            return trade_data_list

    LOGGER.info(f'Loading {market_date} {ticker} trade data...')

    with open(file_path, 'r') as f:
        data_file = csv.DictReader(f)
        for _, row in data_file:
            trade_data = TradeData(
                ticker=ticker,
                trade_price=float(row['Price']),
                trade_volume=float(row['Volume']),
                trade_time=datetime.datetime.combine(market_date, datetime.time(*map(int, row['Time'].split(":")))),
                side=row['Type']
            )
            trade_data.additional['sell_order_id'] = int(row['SaleOrderID'])
            trade_data.additional['buy_order_id'] = int(row['BuyOrderID'])
            trade_data_list.append(trade_data)

    return trade_data_list
