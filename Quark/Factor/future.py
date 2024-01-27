"""
The future model is designed to store dangerous future functions

a future function is a function using future data, which is not available during backtesting.

future functions are designed to be used in prediction / validation.
"""

import datetime

import numpy as np
import pandas as pd

from .decoder import RecursiveDecoder
from ..Base import GlobalStatics

TIME_ZONE = GlobalStatics.TIME_ZONE
__all__ = ['fix_prediction_target', 'wavelet_prediction_target']


def fix_prediction_target(factors: pd.DataFrame, pred_length: float, key: str = 'SyntheticIndex.market_price', inplace: bool = True, session_filter=None) -> pd.DataFrame:
    """
    Given a factor dataframe (StrategyMetrics.info), return the prediction target, with fixed prediction length.

    This function does not take the market (session) breaks into account, as intended.
    And may return a series with Nan values.
    """
    target = dict()
    for ts, row in factors.iterrows():  # type: float, dict
        t0 = ts
        t1 = ts + pred_length

        if session_filter is not None and not session_filter(t0):
            continue

        closest_index = None

        for index in factors.index:
            if index >= t1:
                closest_index = index
                break

        if closest_index is None:
            continue

        # Find the closest index greater or equal to ts + window
        closest_index = factors.index[factors.index >= t1].min()

        # Get the prices at ts and ts + window
        p0 = row[key]
        p1 = factors.at[closest_index, key]

        # Calculate the percentage change and assign it to the 'pct_chg' column
        target[t0] = (p1 / p0) - 1

    if inplace:
        factors['pct_change'] = pd.Series(target).astype(float)
        return factors
    else:
        return pd.DataFrame({'pct_change': pd.Series(target).astype(float)})


def wavelet_prediction_target(factors: pd.DataFrame, key: str = 'SyntheticIndex.market_price', inplace: bool = True, session_filter=None, decoder: RecursiveDecoder = None, decode_level=4, enable_smooth: bool = True, smooth_alpha=0.008, smooth_look_back=5 * 60) -> pd.DataFrame:
    if not inplace:
        factors = pd.DataFrame({key: factors[key]})

    if decoder is None:
        decoder = RecursiveDecoder(level=decode_level)
    decoder.clear()

    # step 1: update decoder
    for _ in factors.iterrows():  # type: tuple[float, dict]
        ts, row = _
        market_price = float(row.get(key, np.nan))
        market_time = datetime.datetime.fromtimestamp(ts, tz=TIME_ZONE)
        timestamp = market_time.timestamp()

        # filter nan values
        if not np.isfinite(market_price):
            continue

        # filter non-trading hours
        if session_filter is not None and not session_filter(ts):
            continue

        decoder.update_decoder(ticker=key, market_price=market_price, timestamp=timestamp)

    # step 2: mark the wave flag (ups and downs) for each data point
    factors['state'] = 0
    local_extreme = decoder.local_extremes(ticker=key, level=decode_level)
    factors['local_max'] = np.nan
    factors['local_min'] = np.nan

    for i in range(len(local_extreme)):
        _, start_ts, flag = local_extreme[i]
        end_ts = local_extreme[i + 1][1] if i + 1 < len(local_extreme) else None

        # Step 1: Reverse the selected DataFrame
        if start_ts is None:
            info_selected = factors.loc[:end_ts][::-1]
        elif end_ts is None:
            info_selected = factors.loc[start_ts:][::-1]
        else:
            info_selected = factors.loc[start_ts:end_ts][::-1]

        # Step 2: Calculate cumulative minimum and maximum of "index_price"
        info_selected['local_max'] = info_selected[key].cummax()
        info_selected['local_min'] = info_selected[key].cummin()
        info_selected['state'] = -flag

        # Step 3: Merge the result back into the original DataFrame
        factors['local_max'].update(info_selected['local_max'])
        factors['local_min'].update(info_selected['local_min'])
        factors['state'].update(info_selected['state'])

    factors['up_actual'] = factors['local_max'] / factors[key] - 1
    factors['down_actual'] = factors['local_min'] / factors[key] - 1
    factors['target_actual'] = np.where(factors['state'] == 1, factors['up_actual'],
                                        np.where(factors['state'] == -1, factors['down_actual'],
                                                 (factors['up_actual'] + factors['down_actual']) / 2))

    # step 3: smooth out the breaking points
    if not enable_smooth:
        return factors

    factors['up_smoothed'] = factors['up_actual']
    factors['down_smoothed'] = factors['down_actual']

    for i in range(len(local_extreme) - 1):
        previous_extreme = local_extreme[i - 1] if i > 0 else None
        break_point = local_extreme[i]
        next_extreme = local_extreme[i + 1]
        next_extreme_price = next_extreme[0]
        break_ts = break_point[1]
        break_type = break_point[2]
        start_ts = max(previous_extreme[1], break_ts - smooth_look_back) if previous_extreme else break_ts - smooth_look_back
        end_ts = break_ts

        smooth_range = factors.loc[start_ts:end_ts]

        if break_type == 1:  # approaching to local maximum, use downward profit is discontinuous, using "up_actual" to smooth out
            max_loss = (-smooth_range.up_actual[::-1]).cummin()[::-1]
            potential = (next_extreme_price / smooth_range[key] - 1).clip(None, 0)
            hold_prob = (-max_loss).apply(lambda _: 1 - _ / smooth_alpha if _ < smooth_alpha else 0)
            smoothed = potential * hold_prob + smooth_range.down_actual * (1 - hold_prob)
            factors['down_smoothed'].update(smoothed)
        elif break_type == -1:
            max_loss = smooth_range.down_actual[::-1].cummin()[::-1]
            potential = (next_extreme_price / smooth_range[key] - 1).clip(0, None)
            hold_prob = (-max_loss).apply(lambda _: 1 - _ / smooth_alpha if _ < smooth_alpha else 0)
            smoothed = potential * hold_prob + smooth_range.up_actual * (1 - hold_prob)
            factors['up_smoothed'].update(smoothed)
        else:
            continue

    factors['target_smoothed'] = np.where(factors['state'] == 1, factors['up_smoothed'],
                                          np.where(factors['state'] == -1, factors['down_smoothed'],
                                                   (factors['up_smoothed'] + factors['down_smoothed']) / 2))

    return factors
