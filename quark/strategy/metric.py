import datetime
import pathlib
import uuid

import numpy as np
import pandas as pd
from algo_engine.base import TradeInstruction, TradeReport

from . import LOGGER
from ..base import GlobalStatics
from ..factor.decoder import Wavelet

LOGGER = LOGGER.getChild('Metrics')


class StrategyMetrics(object):
    def __init__(self, sampling_interval: int):
        self.sampling_interval = sampling_interval

        self.active_entry = {}
        self.factor_value = {}
        self.prediction_value = {}
        self.assets_value = {}

        self.signal_trade_metrics = TradeMetrics()
        self.target_trade_metrics = TradeMetrics()
        self.actual_trade_metrics = TradeMetrics()

        self.assets_price = np.nan
        self.timestamp = 0.

    def collect_synthetic_price(self, synthetic_price: float, timestamp: float):
        self.assets_price = synthetic_price
        timestamp_index = (timestamp // self.sampling_interval) * self.sampling_interval

        if timestamp_index not in self.assets_value:
            self.assets_value[timestamp_index] = {'open': self.assets_price, 'high': self.assets_price, 'low': self.assets_price, 'close': self.assets_price}

        self.assets_value[timestamp_index]['high'] = max(self.assets_price, self.assets_value[timestamp_index]['high'])
        self.assets_value[timestamp_index]['low'] = min(self.assets_price, self.assets_value[timestamp_index]['low'])
        self.assets_value[timestamp_index]['close'] = self.assets_price
        self.timestamp = timestamp

    def collect_factors(self, factor_value: dict[str, float], timestamp: float) -> dict[str, float] | None:
        timestamp_index = (timestamp // self.sampling_interval) * self.sampling_interval

        if not GlobalStatics.PROFILE.is_market_session(timestamp=timestamp):
            return None

        if timestamp_index not in self.factor_value:
            self.factor_value[timestamp_index] = self.active_entry = {}

        self.active_entry.update(factor_value)

        self.timestamp = timestamp
        return factor_value

    def log_wavelet(self, wavelet: Wavelet):
        for ts in self.factor_value:
            if wavelet.start_ts <= ts <= wavelet.end_ts:
                self.factor_value[ts]['Decoder.Flag'] = wavelet.flag.value

    def on_prediction(self, prediction: dict[str, float], timestamp: float):
        self.prediction_value[timestamp] = prediction

    def on_signal(self, signal: int, timestamp: float):
        if signal:
            self.signal_trade_metrics.add_trades(
                side=np.sign(signal),
                price=self.assets_price,
                timestamp=timestamp
            )

    def on_order(self, order: TradeInstruction):
        self.target_trade_metrics.add_trades(
            side=order.side.sign,
            volume=order.volume,
            price=self.assets_price,
            timestamp=self.timestamp,
            trade_id=order.order_id
        )

    def on_trade(self, report: TradeReport):
        self.actual_trade_metrics.add_trades(
            side=report.side.sign,
            volume=report.volume,
            price=report.price,
            timestamp=report.timestamp,
            trade_id=report.trade_id
        )

    def clear(self):
        self.active_entry.clear()
        self.factor_value.clear()
        self.prediction_value.clear()
        self.assets_value.clear()

        self.signal_trade_metrics.clear()
        self.target_trade_metrics.clear()
        self.actual_trade_metrics.clear()

        self.assets_price = np.nan
        self.timestamp = 0.

    def plot_prediction(self):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        prediction_value = pd.DataFrame(self.prediction_value)
        prediction_targets = [pred_var for pred_var in prediction_value.index if not (pred_var.endswith('.lower_bound') or pred_var.endswith('.upper_bound') or pred_var.endswith('.kelly') or pred_var.endswith('.ema'))]
        n = len(prediction_targets)
        titles = ['SyntheticIndex'] + [f'StrategyMetrics: Pred.{pred_var}' for pred_var in prediction_targets]
        fig = make_subplots(
            rows=n + 1,
            cols=1,
            shared_xaxes=True,
            subplot_titles=titles,
            row_heights=[2] + [1] * n
        )

        candle_sticks = self.candle_sticks
        candle_sticks['name'] = 'SyntheticIndex'
        candle_sticks['yaxis'] = 'y'
        # top trace: synthetic candle sticks
        fig.add_trace(
            trace=candle_sticks,
            row=1,
            col=1
        )
        fig.update_layout(
            {f'yaxis': dict(
                title="Synthetic",
                anchor="x",
                side='left',
                showgrid=False,
                showspikes=True,
                spikethickness=-2
            )}
        )

        # trace 2: scatter plot for prediction values
        for i, pred_var in enumerate(prediction_targets):
            prediction = prediction_value.T[pred_var]
            x = np.array([datetime.datetime.fromtimestamp(_, tz=GlobalStatics.TIME_ZONE) for _ in prediction.index])
            y = prediction.to_numpy()

            # trace 2: add prediction value
            fig.add_trace(
                trace=go.Scatter(
                    name=pred_var,
                    x=x,
                    y=y,
                    mode='lines',
                    yaxis=f'y{i + 2}'
                ),
                row=2 + i,
                col=1
            )
            fig.update_layout(
                {f'yaxis{i + 2}': dict(
                    title=pred_var,
                    anchor="x",
                    side='right',
                    showspikes=True,
                    spikethickness=-2,
                    showgrid=True,
                    zeroline=True,
                    showticklabels=True,
                    tickformat='.2%'
                )}
            )

            # trace 2: add prediction interval
            upper_bound = f'{pred_var}.upper_bound'
            lower_bound = f'{pred_var}.lower_bound'
            if upper_bound in prediction_value.index and lower_bound in prediction_value.index:
                y_upper = prediction_value.T[upper_bound].to_numpy()
                y_lower = prediction_value.T[lower_bound].to_numpy()
                valid_mask = np.isfinite(y_upper) & np.isfinite(y_lower)

                fig.add_trace(
                    go.Scatter(
                        name=upper_bound,
                        x=x[valid_mask],
                        y=y_upper[valid_mask],
                        mode='lines',
                        line=dict(color='red', width=1, dash='dot'),
                        yaxis=f'y{i + 2}'
                    ),
                    row=2 + i,
                    col=1
                )

                fig.add_trace(
                    go.Scatter(
                        name=pred_var,
                        x=x[valid_mask],
                        y=y_lower[valid_mask],
                        line=dict(color='red', width=1, dash='dot'),
                        mode='lines',
                        fillcolor='rgba(255,0,0,0.2)',
                        fill='tonexty',
                        yaxis=f'y{i + 2}'
                    ),
                    row=2 + i,
                    col=1
                )

            # trace 3: add ema to the plot
            ema = f'{pred_var}.ema'
            if ema in prediction_value.index:
                y_ema = prediction_value.T[ema].to_numpy()
                fig.add_trace(
                    trace=go.Scatter(
                        name=ema,
                        x=x,
                        y=y_ema,
                        mode='lines',
                        yaxis=f'y{i + 2}'
                    ),
                    row=2 + i,
                    col=1
                )

        fig.update_layout(
            title="StrategyMetrics: Prediction Values",
            height=600 + 300 * n,
            hovermode="x unified",
            template='simple_white',
            showlegend=False
        )

        fig.update_traces(
            xaxis=f'x1'
        )

        fig.update_xaxes(
            tickformat='%H:%M:%S',
            gridcolor='black',
            griddash='dash',
            minor_griddash="dot",
            showgrid=True,
            spikethickness=-2,
            rangebreaks=GlobalStatics.RANGE_BREAK,
            rangeslider_visible=False
        )

        return fig

    def plot_trades(self, max_logs: int = 1500):
        from plotly.subplots import make_subplots

        titles = ['Signal Trades', 'Target Trades', 'Actual Trades']
        trade_metrics = [self.signal_trade_metrics, self.target_trade_metrics, self.actual_trade_metrics]
        n = len(trade_metrics)
        # Create subplot
        fig = make_subplots(
            rows=n,
            cols=1,
            shared_xaxes=True,
            subplot_titles=titles,
        )

        for i, (trade_metrics, metric_name) in enumerate(zip(trade_metrics, titles)):
            trace = self.candle_sticks
            trace['name'] = metric_name
            trace['yaxis'] = f'y{i + 1}'
            fig.add_trace(trace, row=1 + i, col=1)
            fig.update_layout(
                {f'yaxis{i + 1}': dict(
                    anchor="x",
                    side='left',
                    showgrid=False,
                    showspikes=True,
                    spikethickness=-2
                )}
            )
            trade_logs = list(trade_metrics.trades.values())
            if len(trade_logs) > max_logs:
                LOGGER.warning(f'Too many trade logs for {metric_name}, only showing first {max_logs} out of {len(trade_logs)} entries.')

            for trade_dict in trade_logs[:max_logs]:
                action = trade_dict['side']
                x = datetime.datetime.fromtimestamp(trade_dict['timestamp'], tz=GlobalStatics.TIME_ZONE)
                y = trade_dict['price']

                if action > 0:
                    annotation_text = 'Buy'
                    ax, ay, bg_color = 20, -40, 'green'
                elif action < 0:
                    annotation_text = 'Sell'
                    ax, ay, bg_color = -20, 40, 'red'
                else:
                    continue

                fig.add_annotation(
                    x=x,  # x-coordinate
                    y=y,  # y-coordinate (relative to y-axis 1)
                    xref='x',
                    yref=f'y{i + 1}',
                    text=annotation_text,
                    showarrow=True,
                    arrowhead=3,  # red arrow shape
                    ax=ax,  # arrow x-direction offset
                    ay=ay,  # arrow y-direction offset
                    bgcolor=bg_color,
                    opacity=0.5
                )

        fig.update_layout(
            title=dict(text="StrategyMetrics: Trade Signal"),
            height=600 * n,
            template='simple_white',
            showlegend=False,
            hovermode='x unified',
            legend_traceorder="normal"
        )

        fig.update_traces(
            xaxis=f'x1'
        )

        fig.update_xaxes(
            tickformat='%H:%M:%S',
            gridcolor='black',
            griddash='dash',
            minor_griddash="dot",
            showgrid=True,
            spikethickness=-2,
            rangebreaks=GlobalStatics.RANGE_BREAK,
            rangeslider_visible=False
        )

        return fig

    def dump(self, file_path: str | pathlib.Path):
        info = self.info
        # info.index = [datetime.datetime.fromtimestamp(_, tz=GlobalStatics.TIME_ZONE) for _ in info.index]

        info.to_csv(file_path)

    @property
    def info(self) -> pd.DataFrame:
        info = pd.DataFrame(self.factor_value).T
        return info

    @property
    def trade_summary(self) -> pd.DataFrame:
        trade_info = dict(
            signal=self.signal_trade_metrics.summary,
            target=self.target_trade_metrics.summary,
            actual=self.actual_trade_metrics.summary,
        )
        return pd.DataFrame(trade_info)

    @property
    def candle_sticks(self):
        import plotly.graph_objects as go

        trace = go.Candlestick(
            x=[datetime.datetime.fromtimestamp(_, tz=GlobalStatics.TIME_ZONE) for _ in self.assets_value],
            open=[_['open'] for _ in self.assets_value.values()],
            high=[_['high'] for _ in self.assets_value.values()],
            low=[_['low'] for _ in self.assets_value.values()],
            close=[_['close'] for _ in self.assets_value.values()]
        )

        return trace


class TradeMetrics(object):
    def __init__(self):
        self.trades = {}
        self.trade_batch = []

        self.exposure = 0.
        self.total_pnl = 0.
        self.total_cash_flow = 0.

        self._current_pnl = 0.
        self._current_cash_flow = 0.
        self._current_trade_batch = {'cash_flow': 0., 'pnl': 0., 'turnover': 0., 'trades': []}
        self._market_price = None

    def update(self, market_price: float):
        self._market_price = market_price
        self.total_pnl = self.exposure * market_price + self.total_cash_flow
        self._current_pnl = self.exposure * market_price + self._current_cash_flow
        self._current_trade_batch['pnl'] = self.exposure * market_price + self._current_trade_batch['cash_flow']

    def add_trades(self, side: int, price: float, timestamp: float, volume: float = None, trade_id: int | str = None):
        assert side in {1, -1}, f"trade side must in {1, -1}, got {side}."
        assert volume is None or volume >= 0, "volume must be positive."

        if volume is None:
            if self.exposure * side < 0:
                volume = abs(self.exposure)
            elif self.exposure * side > 0:
                volume = 0.
            else:
                volume = 1.

        if trade_id is None:
            trade_id = uuid.uuid4().int
        elif trade_id in self.trades:
            return

        # split the trades
        if (target_exposure := self.exposure + volume * side) * self.exposure < 0:
            self.add_trades(side=side, volume=abs(self.exposure), price=price, timestamp=timestamp, trade_id=f'{trade_id}.0')
            volume = volume - abs(self.exposure)
            trade_id = f'{trade_id}.1'

        self.exposure += volume * side
        self.total_cash_flow -= volume * side * price
        self.total_pnl = self.exposure * price + self.total_cash_flow
        self._current_cash_flow -= volume * side * price
        self._current_pnl = self.exposure * price + self._current_cash_flow
        self._market_price = price

        self.trades[trade_id] = trade_log = dict(
            side=side,
            volume=volume,
            timestamp=timestamp,
            price=price,
            exposure=self.exposure,
            cash_flow=self._current_cash_flow,
            pnl=self._current_pnl
        )

        if 'init_side' not in self._current_trade_batch:
            self._current_trade_batch['init_side'] = side

        self._current_trade_batch['cash_flow'] -= volume * side * price
        self._current_trade_batch['pnl'] = self.exposure * price + self._current_trade_batch['cash_flow']
        self._current_trade_batch['turnover'] += abs(volume) * price
        self._current_trade_batch['trades'].append(trade_log)

        if not self.exposure:
            self.trade_batch.append(self._current_trade_batch)
            self._current_trade_batch = {'cash_flow': 0., 'pnl': 0., 'turnover': 0., 'trades': []}
            self._current_pnl = self._current_cash_flow = 0.

    def add_trades_batch(self, trade_logs: pd.DataFrame):
        for timestamp, row in trade_logs.iterrows():  # type: float, dict
            side = row['side']
            price = row['current_price']
            volume = row['signal']
            self.add_trades(side=side, volume=volume, price=price, timestamp=timestamp)

    def clear(self):
        self.trades.clear()
        self.trade_batch.clear()

        self.exposure = 0.
        self.total_pnl = 0.
        self.total_cash_flow = 0.

        self._current_pnl = 0.
        self._current_cash_flow = 0.
        self._current_trade_batch = {'cash_flow': 0., 'pnl': 0., 'turnover': 0., 'trades': []}
        self._market_price = None

    @property
    def summary(self):
        info_dict = dict(
            total_gain=0.,
            total_loss=0.,
            trade_count=0,
            win_count=0,
            lose_count=0,
            turnover=0.,
        )

        for trade_batch in self.trade_batch:
            if trade_batch['pnl'] > 0:
                info_dict['total_gain'] += trade_batch['pnl']
                info_dict['trade_count'] += 1
                info_dict['win_count'] += 1
                info_dict['turnover'] += trade_batch['turnover']
            else:
                info_dict['total_loss'] += trade_batch['pnl']
                info_dict['trade_count'] += 1
                info_dict['lose_count'] += 1
                info_dict['turnover'] += trade_batch['turnover']

        info_dict['win_rate'] = info_dict['win_count'] / info_dict['trade_count'] if info_dict['trade_count'] else 0.
        info_dict['average_gain'] = info_dict['total_gain'] / info_dict['win_count'] / self._market_price if info_dict['win_count'] else 0.
        info_dict['average_loss'] = info_dict['total_loss'] / info_dict['lose_count'] / self._market_price if info_dict['lose_count'] else 0.
        info_dict['gain_loss_ratio'] = -info_dict['average_gain'] / info_dict['average_loss'] if info_dict['average_loss'] else 1.
        info_dict['long_avg_pnl'] = np.average([_['pnl'] for _ in long_trades]) / self._market_price if (long_trades := [_ for _ in self.trade_batch if _['init_side'] == 1]) else np.nan
        info_dict['short_avg_pnl'] = np.average([_['pnl'] for _ in short_trades]) / self._market_price if (short_trades := [_ for _ in self.trade_batch if _['init_side'] == -1]) else np.nan
        info_dict['ttl_pnl.no_leverage'] = np.sum([trade_batch['pnl'] for trade_batch in self.trade_batch])
        info_dict['net_pnl.optimistic'] = info_dict['ttl_pnl.no_leverage'] - (0.00034 + 0.000023) / 2 * info_dict['turnover']

        return info_dict

    @property
    def info(self):
        trade_info = []
        trade_index = []
        for batch_id, trade_batch in enumerate(self.trade_batch):
            for trade_id, trade_dict in enumerate(trade_batch['trades']):
                trade_info.append(
                    dict(
                        market_time=datetime.datetime.fromtimestamp(trade_dict['timestamp'], tz=GlobalStatics.TIME_ZONE),
                        side=trade_dict['side'],
                        volume=trade_dict['volume'],
                        price=trade_dict['price'],
                        exposure=trade_dict['exposure'],
                        pnl=trade_dict['pnl']
                    )
                )
                trade_index.append((f'batch.{batch_id}', f'trade.{trade_id}'))

        df = pd.DataFrame(trade_info, index=trade_index)
        return df

    def to_string(self) -> str:
        metric_info = self.summary

        fmt_dict = {
            'total_gain': f'{metric_info["total_gain"]:,.3f}',
            'total_loss': f'{metric_info["total_loss"]:,.3f}',
            'trade_count': f'{metric_info["trade_count"]:,}',
            'win_count': f'{metric_info["win_count"]:,}',
            'lose_count': f'{metric_info["lose_count"]:,}',
            'turnover': f'{metric_info["turnover"]:,.3f}',
            'win_rate': f'{metric_info["win_rate"]:.2%}',
            'average_gain': f'{metric_info["average_gain"]:,.4%}',
            'average_loss': f'{metric_info["average_loss"]:,.4%}',
            'long_avg_pnl': f'{metric_info["long_avg_pnl"]:,.4%}',
            'short_avg_pnl': f'{metric_info["short_avg_pnl"]:,.4%}',
            'gain_loss_ratio': f'{metric_info["gain_loss_ratio"]:,.3%}'
        }

        info_str = (f'Trade Metrics Report:'
                    f'\n'
                    f'{pd.Series(fmt_dict).to_string()}'
                    f'\n'
                    f'{self.info.to_string()}')

        return info_str


__all__ = ['StrategyMetrics']
