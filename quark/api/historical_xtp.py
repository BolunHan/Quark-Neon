import csv
import datetime
import os.path
import pathlib
import tarfile
import time
from typing import Iterable

import numpy as np
from algo_engine.base import TransactionSide, TradeData, TransactionData, TickData, OrderBook

from . import LOGGER
from ..base import GlobalStatics

ARCHIVE_DIR = pathlib.Path.home().joinpath('Documents', 'TransactionDataArchive')
DATA_DIR = pathlib.Path.home().joinpath('Documents', 'XTPData')


def extract_archive(market_date: datetime.date, stock_pool: Iterable[str], dtype: str = 'TradeData', **kwargs):
    archive_path = kwargs.get('archive_path', pathlib.Path(ARCHIVE_DIR, f'{market_date:%Y-%m-%d}.tgz'))
    extract_dir = kwargs.get('extract_dir', pathlib.Path(DATA_DIR))
    os.makedirs(extract_dir, exist_ok=True)
    task_components = []
    tasks = []

    if dtype not in (valid_dtype := ['TradeData', 'TransactionData', 'OrderBook', 'TickData', 'All']):
        raise ValueError(f'Expect dtype in {valid_dtype}, got {dtype}!')

    for ticker in set(stock_pool):
        if dtype == 'TradeData':
            task_components.append([f'{market_date:%Y-%m-%d}', 'transactions', f'{ticker}.csv'])
        elif dtype == 'TransactionData':
            task_components.append([f'{market_date:%Y-%m-%d}', 'transactions', f'{ticker}.csv'])
            task_components.append([f'{market_date:%Y-%m-%d}', 'orders', f'{ticker}.csv'])
        elif dtype == 'OrderBook' or dtype == 'TickData':
            task_components.append([f'{market_date:%Y-%m-%d}', 'ticks', f'{ticker}.csv'])
        elif dtype == 'All':
            task_components.append([f'{market_date:%Y-%m-%d}', 'transactions', f'{ticker}.csv'])
            task_components.append([f'{market_date:%Y-%m-%d}', 'orders', f'{ticker}.csv'])
            task_components.append([f'{market_date:%Y-%m-%d}', 'ticks', f'{ticker}.csv'])

    for member_components in task_components:
        extracted_path = pathlib.Path(extract_dir, *member_components)

        if os.path.isfile(extracted_path):
            LOGGER.debug(f'{pathlib.Path(*member_components)} already existed!')
            continue

        member_name = '/'.join(member_components)
        tasks.append(member_name)

    if not tasks:
        LOGGER.info('All tasks already extracted!')
        return

    if not os.path.isfile(archive_path):
        LOGGER.warning(f'Archive {archive_path} not found!')

    with tarfile.open(archive_path, mode='r', bufsize=1024 * 1024) as archive:
        LOGGER.info(f'Getting member info from {archive_path}...')
        all_members: list[tarfile.TarInfo] = archive.getmembers()
        extraction_tasks = []

        for member_name in tasks:
            is_found = False

            for member in all_members:
                if member.name == member_name:
                    extraction_tasks.append(member)
                    LOGGER.info(f'Added {member} to extraction tasks.')
                    is_found = True
                    break

            if not is_found:
                LOGGER.warning(f'{member_name} not found in archive {archive_path}! Skipped!')

        if extraction_tasks:
            LOGGER.info('Extracting...')
            archive.extractall(path=extract_dir, members=extraction_tasks, filter='fully_trusted')
            LOGGER.info('Extraction Complete!')


def load_trade_data(market_date: datetime.date, ticker: str) -> list[TradeData]:
    ts = time.time()
    trade_list = []
    file_path = pathlib.Path(DATA_DIR, f'{market_date:%Y-%m-%d}', 'transactions', f'{ticker}.csv')

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f'{file_path} not found! Use extract_archive prior to this function!')

    with open(file_path, 'r') as f:
        data_file = csv.DictReader(f)
        for row in data_file:  # type: dict[str, str | float]
            price = float(row['price'])
            volume = float(row['volume'])
            notional = float(row['turnover'])
            buy_id = float(row['bid_order_id'])
            sell_id = float(row['ask_order_id'])
            side = TransactionSide.LongFilled if int(row['direction']) == 1 else TransactionSide.ShortFilled if int(row['direction']) == 2 else TransactionSide.CANCEL
            t_str = row['time'].rjust(9, '0')
            market_time = datetime.time(int(t_str[:2]), int(t_str[2:4]), int(t_str[4:6]), int(t_str[6:]))
            timestamp = datetime.datetime.combine(market_date, market_time, tzinfo=GlobalStatics.TIME_ZONE).timestamp()

            trade_data = TradeData(
                ticker=ticker,
                trade_price=price,
                trade_volume=volume,
                notional=notional,
                timestamp=timestamp,
                side=side,
                buy_id=buy_id,
                sell_id=sell_id
            )
            trade_list.append(trade_data)

            if GlobalStatics.DEBUG_MODE:
                if not np.isfinite(trade_data.volume):
                    raise ValueError(f'Invalid trade data {trade_data}, volume = {trade_data.volume}')

                if not np.isfinite(trade_data.price) or trade_data.price < 0:
                    raise ValueError(f'Invalid trade data {trade_data}, price = {trade_data.price}')

                if trade_data.side.value == 0:
                    raise ValueError(f'Invalid trade data {trade_data}, side = {trade_data.side}')

    LOGGER.info(f'{market_date} {ticker} trade data loaded, {len(trade_list):,} entries in {time.time() - ts:.3f}s.')

    return trade_list


def load_transaction_data(market_date: datetime.date, ticker: str) -> list[TransactionData]:
    ts = time.time()
    transaction_list = []
    file_path = pathlib.Path(DATA_DIR, f'{market_date:%Y-%m-%d}', 'orders', f'{ticker}.csv')

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f'{file_path} not found! Use extract_archive prior to this function!')

    with open(file_path, 'r') as f:
        data_file = csv.DictReader(f)
        for row in data_file:  # type: dict[str, str | float]
            price = float(row['price'])
            volume = float(row['volume'])
            side = TransactionSide.BidOrder if int(row['direction']) == 1 else TransactionSide.AskOrder
            transaction_id = int(row['record_id'])
            t_str = row['time'].rjust(9, '0')
            market_time = datetime.time(int(t_str[:2]), int(t_str[2:4]), int(t_str[4:6]), int(t_str[6:]))
            timestamp = datetime.datetime.combine(market_date, market_time, tzinfo=GlobalStatics.TIME_ZONE).timestamp()
            order_kind = int(row['order_kind'])

            # valid order for SZ EX
            if order_kind == 48:
                pass
            # valid order for SH EX
            elif order_kind == 65 or order_kind == 68:
                pass
            else:
                continue

            transaction_data = TransactionData(
                ticker=ticker,
                price=price,
                volume=volume,
                timestamp=timestamp,
                side=side,
                transaction_id=transaction_id
            )

            transaction_list.append(transaction_data)

    LOGGER.info(f'{market_date} {ticker} transaction data loaded, {len(transaction_list):,} entries in {time.time() - ts:.3f}s.')

    return transaction_list


def load_tick_data(market_date: datetime.date, ticker: str) -> list[TickData]:
    ts = time.time()
    tick_list = []
    file_path = pathlib.Path(DATA_DIR, f'{market_date:%Y-%m-%d}', 'ticks', f'{ticker}.csv')

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f'{file_path} not found! Use extract_archive prior to this function!')

    with open(file_path, 'r') as f:
        data_file = csv.DictReader(f)
        for row in data_file:  # type: dict[str, str | float]
            state_code = int(row['status'])
            open_price = float(row['open'])
            high_price = float(row['high'])
            low_price = float(row['low'])
            close_price = float(row['close'])
            pre_close_price = float(row['prev_close'])
            t_str = row['time'].rjust(9, '0')
            market_time = datetime.time(int(t_str[:2]), int(t_str[2:4]), int(t_str[4:6]), int(t_str[6:]))
            timestamp = datetime.datetime.combine(market_date, market_time, tzinfo=GlobalStatics.TIME_ZONE).timestamp()

            limit_up = float(row['limit_up'])
            limit_down = float(row['limit_down'])

            bid_price = float(row['bid_price_1'])
            bid_volume = int(row['bid_volume_1'])
            ask_price = float(row['ask_price_1'])
            ask_volume = int(row['ask_volume_1'])
            total_traded_volume = int(row['volume'])
            total_traded_notional = int(row['turnover'])
            total_trade_count = int(row['num_trades'])

            # state code for session (expect the auction stage) in SZ EX and SH EX
            if state_code == 84:
                pass
            else:
                continue

            bid, ask = [], []

            for i in range(10):
                bid_price = float(row[f'bid_price_{i + 1}'])
                bid_volume = float(row[f'bid_volume_{i + 1}'])
                ask_price = float(row[f'ask_price_{i + 1}'])
                ask_volume = float(row[f'ask_volume_{i + 1}'])

                if bid_price:
                    bid.append([bid_price, bid_volume])

                if ask_price:
                    ask.append([ask_price, ask_volume])

            order_book = OrderBook(
                ticker=ticker,
                timestamp=timestamp,
                bid=bid,
                ask=ask
            )

            tick_data = TickData(
                ticker=ticker,
                timestamp=timestamp,
                last_price=close_price,
                bid_price=bid_price,
                bid_volume=bid_volume,
                ask_price=ask_price,
                ask_volume=ask_volume,
                order_book=order_book,
                total_traded_volume=total_traded_volume,
                total_traded_notional=total_traded_notional,
                total_trade_count=total_trade_count,
                # the rest kwargs is assigned to attribute "additional"
                limit_up=limit_up,
                limit_down=limit_down,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                pre_close_price=pre_close_price
            )

            tick_list.append(tick_data)

    LOGGER.info(f'{market_date} {ticker} tick data loaded, {len(tick_list):,} entries in {time.time() - ts:.3f}s.')

    return tick_list


def loader(market_date: datetime.date, ticker: str, dtype: str):
    if dtype == 'TradeData':
        return load_trade_data(market_date=market_date, ticker=ticker)
    elif dtype == 'TransactionData':
        data = load_trade_data(market_date=market_date, ticker=ticker) + load_transaction_data(market_date=market_date, ticker=ticker)
        data.sort(key=lambda x: x.timestamp)
    elif dtype == 'TickData':
        return load_tick_data(market_date=market_date, ticker=ticker)
    elif dtype == 'OrderBook':
        data = load_tick_data(market_date=market_date, ticker=ticker)
        return [_.level_2 for _ in data if _.level_2 is not None]
    else:
        raise NotImplementedError(f'API.historical does not have a loader function for {dtype}')


def main():
    LOGGER.info('This is a fast test function, should not be called in the production environment.')
    stock_list = ['000001.SZ', '600160.SH']
    market_date = datetime.date(2024, 3, 7)

    extract_archive(market_date=market_date, stock_pool=stock_list, dtype='All')
    for ticker in stock_list:
        trade_list = load_trade_data(ticker=ticker, market_date=market_date)

    # extract_archive(market_date=market_date, stock_pool=stock_list, dtype='TransactionData')
    for ticker in stock_list:
        transaction_list = load_transaction_data(ticker=ticker, market_date=market_date)

    # extract_archive(market_date=market_date, stock_pool=stock_list, dtype='TickData')
    for ticker in stock_list:
        tick_list = load_tick_data(ticker=ticker, market_date=market_date)

    LOGGER.info('all done')


if __name__ == '__main__':
    main()
