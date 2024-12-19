"""
This module provides utilities for generating session dummies and market session masks based on
timestamps. It includes functions to override the default profile settings and generate indicators
for session openings and closings.

The module is primarily used in the context of stock market data, where session times and market hours
are critical for analyzing and modeling trading strategies.

Functions
=========
- profile_cn_override: Overrides the global profile with a specific configuration for the Chinese market.
- session_dummies: Generates indicators for session openings and closings based on timestamps.
- is_market_session: Checks whether a given timestamp falls within the market session.
- market_session_mask: Generates a boolean mask indicating whether each timestamp in a list falls within the market session.

Constants
=========
- DUMMIES_SESSION_OPENING: Default opening time for session dummies (00:00).
- DUMMIES_SESSION_CLOSING: Default closing time for session dummies (23:59:59.999999).
- RANGE_BREAK: The range break time defined in the global profile.
- TIME_ZONE: The time zone used for timestamp conversion.
"""
import datetime

import pandas as pd

from ..base import GlobalStatics

RANGE_BREAK = GlobalStatics.RANGE_BREAK
TIME_ZONE = GlobalStatics.TIME_ZONE

# The following dummy marks the session opening and closing times for inputs:
# "Dummies.IsOpening" and "Dummies.IsClosing".
# Default is min and max, which means no mask for session opening and closing.
DUMMIES_SESSION_OPENING = datetime.time.min
DUMMIES_SESSION_CLOSING = datetime.time.max


def session_dummies(timestamp: float | list[float], inplace: dict[str, float | list[float]] | pd.DataFrame = None) -> dict:
    """
    Generate session dummies indicating whether the timestamp corresponds to the opening or closing time.

    Args:
        timestamp (float or list[float]): Unix timestamp or a list of Unix timestamps.
        inplace (dict[str, float or list[float]] or pd.DataFrame, optional): Optional dictionary or DataFrame to update in place.

    Returns:
        dict: A dictionary with keys 'Dummies.IsOpening' and 'Dummies.IsClosing', where the values are
        indicators (0 or 1) of whether the timestamp falls before the session opening time or after the session
        closing time, respectively.
    """
    d = {} if inplace is None else inplace

    if isinstance(timestamp, (float, int)):
        # For a single timestamp
        d['Dummies.IsOpening'] = 1 if datetime.datetime.fromtimestamp(timestamp, tz=GlobalStatics.TIME_ZONE).time() < DUMMIES_SESSION_OPENING else 0
        d['Dummies.IsClosing'] = 1 if datetime.datetime.fromtimestamp(timestamp, tz=GlobalStatics.TIME_ZONE).time() > DUMMIES_SESSION_CLOSING else 0
    else:
        # For a list of timestamps
        d['Dummies.IsOpening'] = [1 if datetime.datetime.fromtimestamp(_, tz=GlobalStatics.TIME_ZONE).time() < DUMMIES_SESSION_OPENING else 0 for _ in timestamp]
        d['Dummies.IsClosing'] = [1 if datetime.datetime.fromtimestamp(_, tz=GlobalStatics.TIME_ZONE).time() > DUMMIES_SESSION_CLOSING else 0 for _ in timestamp]

    return d


def is_market_session(timestamp: float | int) -> bool:
    """
    Check whether a given timestamp falls within the market session.

    Args:
        timestamp (float or int): Unix timestamp.

    Returns:
        bool: True if the timestamp falls within the market session, False otherwise.
    """
    return GlobalStatics.PROFILE.is_market_session(timestamp=timestamp)


def market_session_mask(timestamp: list[float], inplace: dict[str, float | list[float]] | pd.DataFrame = None) -> list[bool]:
    """
    Generate a boolean mask indicating whether each timestamp in a list falls within the market session.

    Args:
        timestamp (list[float]): List of Unix timestamps.
        inplace (dict[str, float or list[float]] or pd.DataFrame, optional): Optional dictionary or DataFrame to update in place.

    Returns:
        list[bool]: A list of booleans where True indicates that the timestamp is within the market session and False otherwise.
    """
    mask = [is_market_session(_) for _ in timestamp]

    if inplace:
        inplace['is_valid'] = mask

    return mask


from .cn import profile_cn_override
