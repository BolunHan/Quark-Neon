import datetime
import pathlib
import uuid

import numpy as np
import pandas as pd
from PyQuantKit import TradeInstruction, TradeReport

from ..Base import GlobalStatics
from ..Calibration.dummies import is_market_session
from ..Factor import collect_factor, FactorMonitor
from ..Factor.decoder import Wavelet

TIME_ZONE = GlobalStatics.TIME_ZONE
RANGE_BREAK = GlobalStatics.RANGE_BREAK


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
        self.assets_price = factor_value.get(f'SyntheticIndex.market_price')
        timestamp_index = (timestamp // self.sampling_interval) * self.sampling_interval

        if not is_market_session(timestamp=timestamp):
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
        self.signal_trade_metrics.add_trades(
            volume=signal,
            price=self.assets_price,
            timestamp=timestamp
        )

    def on_order(self, order: TradeInstruction):
        self.target_trade_metrics.add_trades(
            volume=order.volume * order.side.sign,
            price=self.assets_price,
            timestamp=self.timestamp,
            trade_id=order.order_id
        )

    def on_trade(self, report: TradeReport):
        self.actual_trade_metrics.add_trades(
            volume=report.volume * report.side.sign,
            price=report.price,
            timestamp=report.timestamp,
            trade_id=report.trade_id
        )

    def clear(self):
        self.factor_value.clear()
        self.assets_value.clear()

        self.signal_trade_metrics.clear()
        self.target_trade_metrics.clear()
        self.actual_trade_metrics.clear()

        self.assets_price = np.nan
        self.timestamp = 0.

    def plot_prediction(self):
        import plotly.graph_objects as go
        fig = go.Figure()

        # trace 1: candle stick for assets price
        fig.add_trace(
            go.Candlestick(
                name='SyntheticIndex',
                x=[datetime.datetime.fromtimestamp(_, tz=TIME_ZONE) for _ in self.assets_value],
                open=[_['open'] for _ in self.assets_value.values()],
                high=[_['high'] for _ in self.assets_value.values()],
                low=[_['low'] for _ in self.assets_value.values()],
                close=[_['close'] for _ in self.assets_value.values()],
                yaxis='y1'
            )
        )

        # trace 2: scatter plot for prediction values
        prediction_value = pd.DataFrame(self.prediction_value)
        for _, prediction in prediction_value.iterrows():
            x = [datetime.datetime.fromtimestamp(_, tz=TIME_ZONE) for _ in prediction.index]
            y = prediction.tolist()
            # noinspection PyTypeChecker
            pred_var: str = prediction.name

            if not (pred_var.endswith('.lower_bound') or pred_var.endswith('.upper_bound') or pred_var.endswith('.kelly')):
                fig.add_trace(
                    go.Scatter(
                        name=pred_var,
                        x=x,
                        y=y,
                        mode='lines',
                        yaxis='y2'
                    )
                )

                if (upper_bound := f'{pred_var}.upper_bound') in prediction_value.index:
                    y = prediction_value.T[upper_bound].tolist()
                    fig.add_trace(
                        go.Scatter(
                            name=upper_bound,
                            x=x,
                            y=y,
                            mode='lines',
                            line=dict(color='red', dash='dash'),
                            yaxis='y2'
                        )
                    )

                if (lower_bound := f'{pred_var}.lower_bound') in prediction_value.index:
                    y = prediction_value.T[lower_bound].tolist()
                    fig.add_trace(
                        go.Scatter(
                            name=pred_var,
                            x=x,
                            y=y,
                            line=dict(color='red', dash='dash'),
                            mode='lines',
                            fillcolor='rgba(255,0,0,0.3)',
                            fill='tonexty',
                            yaxis='y2'
                        )
                    )

                if (kelly := f'{pred_var}.kelly') in prediction_value.index:
                    y = prediction_value.T[kelly].tolist()
                    fig.add_trace(
                        go.Bar(
                            name=kelly,
                            x=x,
                            y=y,
                            opacity=0.3,
                            marker=dict(color='green'),
                            yaxis='y3'
                        )
                    )

        fig.update_layout(
            title="StrategyMetrics: Prediction",
            hovermode="x unified",
            template='simple_white',
            yaxis=dict(
                title="Synthetic",
                anchor="x",
                overlaying='y',
                side='right',
                showgrid=False
            ),
            yaxis2=dict(
                showspikes=True
            ),
            yaxis3=dict(
                anchor="free",
                overlaying='y',
                showgrid=False,  # Hide grid for y3 axis
                showline=False,  # Hide line for y3 axis
                zeroline=False,  # Hide zero line for y3 axis
                showticklabels=False  # Hide tick labels for y3 axis
            )
        )

        fig.update_xaxes(
            tickformat='%H:%M:%S',
            gridcolor='black',
            griddash='dash',
            minor_griddash="dot",
            showgrid=True,
            spikethickness=-2,
            rangebreaks=RANGE_BREAK,
            rangeslider_visible=False
        )

        return fig

    def plot_trades(self):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        titles = ['Signal', 'Target', 'Actual']
        trade_metrics = [self.signal_trade_metrics, self.target_trade_metrics, self.actual_trade_metrics]
        # Create subplot
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=False,
            subplot_titles=titles,
        )

        for i, (trade_metrics, metric_name) in enumerate(zip(trade_metrics, titles)):
            trace = go.Candlestick(
                name=metric_name,
                x=[datetime.datetime.fromtimestamp(_, tz=TIME_ZONE) for _ in self.assets_value],
                open=[_['open'] for _ in self.assets_value.values()],
                high=[_['high'] for _ in self.assets_value.values()],
                low=[_['low'] for _ in self.assets_value.values()],
                close=[_['close'] for _ in self.assets_value.values()],
                yaxis=f'y{i + 1}'
            )
            fig.add_trace(trace, row=1 + i, col=1)

            for trade_dict in trade_metrics.trades.values():
                fig.add_annotation(
                    x=datetime.datetime.fromtimestamp(trade_dict['timestamp'], tz=TIME_ZONE),  # x-coordinate
                    y=trade_dict['price'],  # y-coordinate (relative to y-axis 1)
                    xref='x',
                    yref=f'y{i + 1}',
                    text='Sell' if trade_dict['volume'] < 0 else 'Buy',
                    showarrow=True,
                    arrowhead=3,  # red arrow shape
                    # ax=20 if action < 0 else -20,  # arrow x-direction offset
                    # ay=-40,  # arrow y-direction offset
                    bgcolor='red' if trade_dict['volume'] < 0 else 'green',
                    opacity=0.8
                )

        fig.update_layout(
            title=dict(text="StrategyMetrics: Trade Signal"),
            height=600 * 3,
            template='simple_white',
            # legend_tracegroupgap=330,
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
            rangebreaks=RANGE_BREAK,
            rangeslider_visible=False
        )

        return fig

    def dump(self, file_path: str | pathlib.Path):
        info = self.info
        # info.index = [datetime.datetime.fromtimestamp(_, tz=TIME_ZONE) for _ in info.index]

        info.to_csv(file_path)

    @property
    def info(self) -> pd.DataFrame:
        info = pd.DataFrame(self.factor_value).T
        return info

    @property
    def trade_info(self) -> pd.DataFrame:
        trade_info = dict(
            signal=self.signal_trade_metrics.info,
            target=self.target_trade_metrics.info,
            actual=self.actual_trade_metrics.info,
        )
        return pd.DataFrame(trade_info)


class TradeMetrics(object):
    def __init__(self):
        self.trades = {}
        self.trade_batch = []

        self.exposure = 0.
        self.cash_flow = 0.
        self.pnl = 0.

        self.current_trade_batch = {}
        self.current_price = None

    def add_trades(self, volume: float, price: float, timestamp: float, trade_id: int | str = None):
        if not volume:
            return

        self.exposure += volume
        self.cash_flow -= volume * price
        self.pnl = self.exposure * price + self.cash_flow
        self.current_price = price

        if trade_id is None:
            trade_id = uuid.uuid4().int
        elif trade_id in self.trades:
            return

        self.trades[trade_id] = dict(
            timestamp=timestamp,
            volume=volume,
            price=price,
            exposure=self.exposure,
            cash_flow=self.cash_flow,
            pnl=self.pnl
        )

        if 'init_side' not in self.current_trade_batch:
            self.current_trade_batch['init_side'] = 1 if volume > 0 else -1

        self.current_trade_batch['cash_flow'] = self.current_trade_batch.get('cash_flow', 0.) - volume * price
        self.current_trade_batch['pnl'] = self.current_trade_batch.get('pnl', 0.) + self.exposure * price + self.current_trade_batch['cash_flow']
        self.current_trade_batch['turnover'] = self.current_trade_batch.get('turnover', 0.) + abs(volume) * price

        if not self.exposure:
            self.trade_batch.append(self.current_trade_batch)
            self.current_trade_batch = {}

    def add_trades_batch(self, trade_logs: pd.DataFrame):
        for timestamp, row in trade_logs.iterrows():  # type: float, dict
            price = row['current_price']
            volume = row['signal']
            self.add_trades(volume=volume, price=price, timestamp=timestamp)

    def clear(self):
        self.trades.clear()
        self.trade_batch.clear()
        self.exposure = 0.
        self.cash_flow = 0.
        self.pnl = 0.
        self.current_trade_batch.clear()
        self.current_price = None

    @property
    def info(self):
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
        info_dict['average_gain'] = info_dict['total_gain'] / info_dict['win_count'] / self.current_price if info_dict['win_count'] else 0.
        info_dict['average_loss'] = info_dict['total_loss'] / info_dict['lose_count'] / self.current_price if info_dict['lose_count'] else 0.
        info_dict['gain_loss_ratio'] = -info_dict['average_gain'] / info_dict['average_loss'] if info_dict['average_loss'] else 1.
        info_dict['long_avg_pnl'] = np.average([_['pnl'] for _ in long_trades]) / self.current_price if (long_trades := [_ for _ in self.trade_batch if _['init_side'] == 1]) else np.nan
        info_dict['short_avg_pnl'] = np.average([_['pnl'] for _ in short_trades]) / self.current_price if (short_trades := [_ for _ in self.trade_batch if _['init_side'] == -1]) else np.nan

        return info_dict

    def to_string(self) -> str:
        metric_info = self.info

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

        return 'Trade Metrics Report:\n' + pd.Series(fmt_dict).to_string()


__all__ = ['StrategyMetrics']
