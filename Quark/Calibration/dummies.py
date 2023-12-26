import datetime
import pandas as pd

from ..Base import GlobalStatics

TIME_ZONE = GlobalStatics.TIME_ZONE


def session_dummies(timestamp: float | list[float], inplace: dict[str, float | list[float]] | pd.DataFrame = None) -> dict:
    """
    Generate session dummies indicating whether the timestamp corresponds to the opening or closing time.

    Args:
        timestamp (float or list[float]): Unix timestamp or a list of Unix timestamps.
        inplace (dict[str, float or list[float]] or pd.DataFrame, optional): Optional dictionary or DataFrame to update in place.

    Returns:
        dict: Dictionary with keys 'Dummies.IsOpening' and 'Dummies.IsClosing'.
    """
    d = {} if inplace is None else inplace

    if isinstance(timestamp, (float, int)):
        # For a single timestamp
        d['Dummies.IsOpening'] = 1 if datetime.datetime.fromtimestamp(timestamp, tz=TIME_ZONE).time() < datetime.time(10, 30) else 0
        d['Dummies.IsClosing'] = 1 if datetime.datetime.fromtimestamp(timestamp, tz=TIME_ZONE).time() > datetime.time(14, 30) else 0
    else:
        # For a list of timestamps
        d['Dummies.IsOpening'] = [1 if datetime.datetime.fromtimestamp(_, tz=TIME_ZONE).time() < datetime.time(10, 30) else 0 for _ in timestamp]
        d['Dummies.IsClosing'] = [1 if datetime.datetime.fromtimestamp(_, tz=TIME_ZONE).time() > datetime.time(14, 30) else 0 for _ in timestamp]

    return d


def is_cn_market_session(timestamp: float | list[float], inplace: dict[str, float | list[float]] | pd.DataFrame = None) -> dict:
    """
    Filter non-trading hours based on the Chinese market session.

    Args:
        timestamp (float or list[float]): Unix timestamp or a list of Unix timestamps.
        inplace (dict[str, float or list[float]] or pd.DataFrame, optional): Optional dictionary or DataFrame to update in place.

    Returns:
        dict: Dictionary with a key 'is_valid', indicating whether the timestamp is within trading hours.
    """
    d = {} if inplace is None else inplace

    if isinstance(timestamp, (float, int)):
        # For a single timestamp
        market_time = datetime.datetime.fromtimestamp(timestamp, tz=TIME_ZONE).time()
        d['is_valid'] = 0 if market_time < datetime.time(9, 30) or datetime.time(11, 30) < market_time < datetime.time(13, 0) or datetime.time(15, 0) < market_time else 1
    else:
        # For a list of timestamps
        d['is_valid'] = [
            0 if ((market_time := datetime.datetime.fromtimestamp(_, tz=TIME_ZONE).time()) < datetime.time(9, 30)
                  or datetime.time(11, 30) < market_time < datetime.time(13, 0)
                  or datetime.time(15, 0) < market_time)
            else 1 for _ in timestamp
        ]

    return d
