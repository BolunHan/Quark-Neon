"""
The future model is designed to store dangerous future functions

a future function is a function using future data, which is not available during backtesting.

future functions are designed to be used in prediction / validation.
"""

import datetime

import numpy as np
import pandas as pd

from .decoder import RecursiveDecoder
from ..base import GlobalStatics

__all__ = ['fix_prediction_target', 'wavelet_prediction_target']


def fix_prediction_target(factors: pd.DataFrame, pred_length: float, key: str = 'SyntheticIndex.market_price', inplace: bool = True, session_filter=None) -> pd.DataFrame:
    """
    Given a factor dataframe (StrategyMetrics.info), return the prediction target, with fixed prediction length.

    This function does not take the market (session) breaks into account, as intended.
    And may return a series with Nan values.
    """
    DeprecationWarning('Use quark.calibration.future.PredictionTarget instead!')
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


def wavelet_prediction_target(factors: pd.DataFrame, key: str = 'SyntheticIndex.market_price', inplace: bool = True, session_filter=None, decoder: RecursiveDecoder = None, decode_level=4, enable_smooth: bool = True, smooth_alpha=0.008, smooth_look_back=15 * 60) -> pd.DataFrame:
    DeprecationWarning('Use quark.calibration.future.PredictionTarget instead!')
    if not inplace:
        factors = pd.DataFrame({key: factors[key]})

    if decoder is None:
        decoder = RecursiveDecoder(level=decode_level)
    decoder.clear()

    # step 1: update decoder
    for _ in factors.iterrows():  # type: tuple[float, dict]
        ts, row = _
        market_price = float(row.get(key, np.nan))
        market_time = datetime.datetime.fromtimestamp(ts, tz=GlobalStatics.TIME_ZONE)
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

    for (_, start_ts, flag), (_, end_ts, _) in zip(local_extreme, local_extreme[1:] + [(None, factors.index[-1], None)]):
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
        # updated to compliant with (pandas) CoW rules
        # factors['local_max'].update(info_selected['local_max'])
        # factors['local_min'].update(info_selected['local_min'])
        # factors['state'].update(info_selected['state'])
        factors.update(info_selected[['local_max', 'local_min', 'state']])

    factors['up_actual'] = factors['local_max'] / factors[key] - 1
    factors['down_actual'] = factors['local_min'] / factors[key] - 1
    factors['target_actual'] = np.where(factors['state'] == 1, factors['up_actual'],
                                        np.where(factors['state'] == -1, factors['down_actual'],
                                                 (factors['up_actual'] + factors['down_actual'])))

    # step 3: smooth out the breaking points
    if not enable_smooth:
        return factors

    factors['up_smoothed'] = factors['up_actual']
    factors['down_smoothed'] = factors['down_actual']
    factors['target_smoothed'] = factors['target_actual']

    for previous_extreme, current_extreme, next_extreme in zip(local_extreme, local_extreme[1:], local_extreme[2:]):
        _, previous_ts, _ = previous_extreme
        break_price, break_ts, break_type = current_extreme
        next_extreme_price, _, _ = next_extreme

        start_ts = max(break_ts - smooth_look_back, previous_ts)
        smooth_range = factors.loc[start_ts:break_ts]

        # approaching a local minimum break
        if break_type == -1:
            potential_loss = smooth_range.down_actual[::-1].cummin()[::-1]
            potential_gain = (next_extreme_price / smooth_range[key] - 1).clip(0, None)
            max_loss = min(potential_loss)
            smooth_threshold = max(max_loss, -smooth_alpha)
            hold_prob = potential_loss.apply(lambda _: 1 - _ / smooth_threshold if _ > smooth_threshold else 0)
            up_smoothed = hold_prob * potential_gain + (1 - hold_prob) * smooth_range.up_actual
            target_smoothed = hold_prob * potential_gain + (1 - hold_prob) * smooth_range.down_smoothed
            # factors['up_smoothed'].update(up_smoothed)
            # factors['target_smoothed'].update(target_smoothed)
            factors.update({'up_smoothed': up_smoothed, 'target_smoothed': target_smoothed})
        elif break_type == 1:
            potential_loss = (-smooth_range.up_actual[::-1]).cummin()[::-1]
            potential_gain = (next_extreme_price / smooth_range[key] - 1).clip(None, 0)  # in this case, the potential gain is negative,
            max_loss = min(potential_loss)
            smooth_threshold = max(max_loss, -smooth_alpha)
            hold_prob = potential_loss.apply(lambda _: 1 - _ / smooth_threshold if _ > smooth_threshold else 0)
            down_smoothed = hold_prob * potential_gain + (1 - hold_prob) * smooth_range.down_actual
            target_smoothed = hold_prob * potential_gain + (1 - hold_prob) * smooth_range.up_smoothed
            # factors['down_smoothed'].update(down_smoothed)
            # factors['target_smoothed'].update(target_smoothed)
            factors.update({'down_smoothed': down_smoothed, 'target_smoothed': target_smoothed})
        else:
            continue

    return factors
