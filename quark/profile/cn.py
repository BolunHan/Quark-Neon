import datetime

import pandas as pd
from algo_engine.profile import PROFILE_CN

from ..base import GlobalStatics

RANGE_BREAK = GlobalStatics.RANGE_BREAK
TIME_ZONE = GlobalStatics.TIME_ZONE

# the following dummy marks the dummies for inputs: "Dummies.IsOpening" and "Dummies.IsClosing",
DUMMIES_SESSION_OPENING = datetime.time(10, 0)
DUMMIES_SESSION_CLOSING = datetime.time(14, 30)


def profile_cn_override():
    PROFILE_CN.override_profile(GlobalStatics.PROFILE)


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
        d['Dummies.IsOpening'] = 1 if datetime.datetime.fromtimestamp(timestamp, tz=TIME_ZONE).time() < DUMMIES_SESSION_OPENING else 0
        d['Dummies.IsClosing'] = 1 if datetime.datetime.fromtimestamp(timestamp, tz=TIME_ZONE).time() > DUMMIES_SESSION_CLOSING else 0
    else:
        # For a list of timestamps
        d['Dummies.IsOpening'] = [1 if datetime.datetime.fromtimestamp(_, tz=TIME_ZONE).time() < DUMMIES_SESSION_OPENING else 0 for _ in timestamp]
        d['Dummies.IsClosing'] = [1 if datetime.datetime.fromtimestamp(_, tz=TIME_ZONE).time() > DUMMIES_SESSION_CLOSING else 0 for _ in timestamp]

    return d


def is_market_session(timestamp: float | int) -> bool:
    return GlobalStatics.PROFILE.is_market_session(timestamp=timestamp)


def market_session_mask(timestamp: list[float], inplace: dict[str, float | list[float]] | pd.DataFrame = None) -> list[bool]:
    mask = [is_market_session(_) for _ in timestamp]

    if inplace:
        inplace['is_valid'] = mask

    return mask
