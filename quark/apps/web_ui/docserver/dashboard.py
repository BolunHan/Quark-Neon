import ctypes
import datetime
from collections.abc import Iterable
from functools import partial
from typing import TypedDict

import numpy as np
from algo_engine.apps import DocServer, DocTheme
from algo_engine.base import MarketData, TradeData, TransactionData
from algo_engine.profile import Profile, PROFILE
from algo_engine.utils import ts_indices
from bokeh.models import PanTool, WheelPanTool, WheelZoomTool, BoxZoomTool, ResetTool, ExamineTool, SaveTool, CrosshairTool, HoverTool, RangeTool, Range1d, ColumnDataSource, FixedTicker, Span, NumeralTickFormatter, SingleIntervalTicker
from bokeh.plotting import figure, gridplot
from colorcet import glasbey_light

from . import LOGGER
from ....calibration.future import FutureTopic


class CandlestickBuffer(ctypes.Structure):
    _fields_ = [
        ('ver', ctypes.c_uint),
        ('ts_start', ctypes.c_double),
        ('ts_end', ctypes.c_double),
        ('open_price', ctypes.c_double),
        ('close_price', ctypes.c_double),
        ('high_price', ctypes.c_double),
        ('low_price', ctypes.c_double),
        ('volume', ctypes.c_double)
    ]


class PredictionBuffer(ctypes.Structure):
    _fields_ = [
        ('ver', ctypes.c_uint),
        ('value', ctypes.c_double),
        ('upper_bound', ctypes.c_double),
        ('lower_bound', ctypes.c_double)
    ]


class DashboardTheme(DocTheme):
    stick_padding = 0.1
    range_padding = 0.01

    ColorStyle = TypedDict('ColorStyle', fields={'up': str, 'down': str})
    ws_style = ColorStyle(up="green", down="red")
    cn_style = ColorStyle(up="red", down="green")
    num_format = '{0,0.00}'
    pct_format = '{0,0.00%}'
    time_format = '{%H:%M:%S}'

    def __init__(self, profile: Profile = PROFILE, style: ColorStyle = None):
        self.profile = profile

        if style is None:
            if profile.profile_id == 'cn':
                self.style = self.cn_style
            else:
                self.style = self.ws_style
        else:
            self.style = style

    def stick_style(self, pct_change: float | int) -> dict:
        style_dict = dict()

        if pct_change > 0:
            style_dict['stick_color'] = self.style['up']
        else:
            style_dict['stick_color'] = self.style['down']

        return style_dict


class Dashboard(DocServer):
    def __init__(self, ticker: str, start_date: datetime.date, end_date: datetime.date, pred_var: str | list[str] | FutureTopic | list[FutureTopic] = None, profile: Profile = PROFILE, interval: float = 60., x_axis: list[float] = None, theme: DocTheme = None, **kwargs):
        self.ticker = ticker
        self.pred_var = [pred_var] if isinstance(pred_var, (str, FutureTopic)) else [] if pred_var is None else pred_var
        self.start_date = start_date
        self.end_date = end_date
        self.profile = profile
        self.interval = interval
        self.indices = self.ts_indices() if x_axis is None else x_axis

        assert self.indices, 'Must assign x_axis to render candlesticks!'

        super().__init__(
            theme=theme,
            max_size=kwargs.get('max_size'),
            update_interval=kwargs.get('update_interval', 0),
        )

        self.theme = DashboardTheme(profile=self.profile) if self.theme is None else self.theme
        self.timestamp: float = 0.
        self.candlestick_buffer = (CandlestickBuffer * len(self.indices))()
        self.prediction_buffer = ((PredictionBuffer * len(self.pred_var)) * len(self.indices))()
        self.last_idx = 0
        self.last_pred = {pred_var: {} for pred_var in self.pred_var}
        self.last_update_ts = 0.

    def ts_indices(self) -> list[float]:
        """generate integer indices
        from start date to end date, with given interval, in seconds
        """

        calendar = self.profile.trade_calendar(start_date=self.start_date, end_date=self.end_date)
        timestamps = []
        for market_date in calendar:
            _ts_indices = ts_indices(
                market_date=market_date,
                interval=self.interval,
                session_start=self.profile.session_start,
                session_end=self.profile.session_end,
                session_break=self.profile.session_break,
                time_zone=self.profile.time_zone,
                ts_mode='both'
            )

            timestamps.extend(_ts_indices)

        return timestamps

    def loc_indices(self, timestamp: float, start_idx: int = 0) -> tuple[int, float]:
        last_idx = idx = max(start_idx - 1, 0)

        if not self.indices:
            raise ValueError('Timestamp indices not initialized!')

        if start_idx >= len(self.indices):
            LOGGER.warning('Start_idx out of range! Resetting the searching index!')
            idx = 0

        if timestamp < self.indices[idx]:
            LOGGER.warning(f'Cached timestamp index {idx} {self.indices[idx]} greater than queried timestamp {timestamp}! Resetting the searching index!')
            idx = 0

        while idx < len(self.indices):
            ts = self.indices[idx]

            if ts > timestamp:
                break

            last_idx = idx
            idx += 1

        return last_idx, self.indices[last_idx]

    def update(self, **kwargs):
        kwargs = kwargs.copy()
        is_updated = False

        if 'timestamp' in kwargs:
            # LOGGER.info(f'{self} timestamp updated {self.timestamp}')
            self.timestamp = kwargs['timestamp']

        # step 1: update candlesticks
        if 'market_data' in kwargs:
            market_data: MarketData = kwargs['market_data']

            if market_data.ticker != self.ticker:
                return

            if isinstance(market_data, (TradeData, TransactionData)):
                self._update_candlestick(timestamp=market_data.timestamp, market_price=market_data.price, volume=market_data.volume)
            else:
                self._update_candlestick(timestamp=market_data.timestamp, market_price=market_data.market_price)
            self.timestamp = market_data.timestamp
            is_updated = True
        elif 'ticker' in kwargs and 'market_price' in kwargs:
            ticker = kwargs['ticker']
            market_price = kwargs.get('close_price', kwargs.get('market_price'))
            open_price = kwargs.get('open_price')
            high_price = kwargs.get('high_price')
            low_price = kwargs.get('low_price')
            volume = kwargs.get('volume', 0)

            assert ticker is not None, 'Must assign a ticker for update function!'
            assert market_price is not None, f'Must assign a market_price or close_price for {self.__class__} update function!'

            if ticker != self.ticker:
                return

            self._update_candlestick(
                timestamp=self.timestamp,
                market_price=market_price,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                volume=volume
            )
            is_updated = True

        # step 2: update prediction
        for pred_var in self.pred_var:
            if pred_var in kwargs:
                value = kwargs[pred_var]
                upper_bound = kwargs.get(f'{pred_var}.upper_bound', pred_var)
                lower_bound = kwargs.get(f'{pred_var}.lower_bound', pred_var)

                if np.isfinite(value) and np.isfinite(upper_bound) and np.isfinite(lower_bound):
                    is_updated = True
                    self._update_prediction(
                        timestamp=self.timestamp,
                        pred_var=pred_var,
                        value=value,
                        upper_bound=upper_bound,
                        lower_bound=lower_bound
                    )
                    self.last_pred[pred_var].update({
                        f'{pred_var}.value': value,
                        f'{pred_var}.upper_bound': upper_bound,
                        f'{pred_var}.lower_bound': lower_bound
                    })

        if is_updated and (self.last_update_ts + self.update_interval < self.timestamp):
            for doc_id in list(self.bokeh_documents):
                doc = self.bokeh_documents[doc_id]
                self._update_datasource(doc_id=doc_id)

                if not self.update_interval:
                    doc.add_next_tick_callback(partial(self.stream, doc_id=doc_id))
                    doc.add_next_tick_callback(partial(self.patch, doc_id=doc_id))

            self.last_update_ts = (self.timestamp // self.update_interval) * self.update_interval if self.update_interval else self.timestamp

    def _update_candlestick(self, timestamp: float, market_price: float, open_price: float = None, high_price: float = None, low_price: float = None, volume: float = 0.) -> CandlestickBuffer:
        if open_price is None:
            open_price = market_price

        if high_price is None:
            high_price = market_price

        if low_price is None:
            low_price = market_price

        int_idx, ts_idx = self.loc_indices(timestamp=timestamp, start_idx=self.last_idx)
        candlestick: CandlestickBuffer = self.candlestick_buffer[int_idx]

        if candlestick.ver:
            candlestick.high_price = max(high_price, candlestick.high_price)
            candlestick.low_price = min(low_price, candlestick.low_price)
            candlestick.close_price = market_price
        else:
            candlestick.high_price = high_price
            candlestick.low_price = low_price
            candlestick.close_price = market_price
            candlestick.open_price = open_price
            candlestick.ts_start = ts_idx
            candlestick.ts_end = ts_idx + self.interval

        candlestick.ver += 1

        if volume:
            candlestick.volume += volume

        return candlestick

    def _update_prediction(self, timestamp: float, pred_var: str | FutureTopic, value: float, upper_bound: float = None, lower_bound: float = None) -> PredictionBuffer:
        pred_idx = self.pred_var.index(pred_var)
        int_idx, ts_idx = self.loc_indices(timestamp=timestamp, start_idx=self.last_idx)

        prediction_buffer = self.prediction_buffer[int_idx][pred_idx]

        prediction_buffer.ver += 1
        prediction_buffer.value = value
        prediction_buffer.upper_bound = upper_bound
        prediction_buffer.lower_bound = lower_bound

        return prediction_buffer

    def _update_datasource(self, doc_id: int = None) -> tuple[dict[str, list], dict[str, list[tuple]], dict[str, list]]:
        if doc_id is not None:
            data_pipe = self.bokeh_data_pipe[doc_id]
            data_patch = self.bokeh_data_patch[doc_id]
            data_source = self.bokeh_data_source[doc_id].data
        else:
            data_pipe = {
                'index': [],  # the plotting index
                'market_time': [],
                'candlestick.ver': [],
                'candlestick.open_price': [],
                'candlestick.high_price': [],
                'candlestick.low_price': [],
                'candlestick.close_price': [],
                'candlestick.volume': [],
                'candlestick.upper_bound': [],
                'candlestick.lower_bound': [],
                'candlestick.color': [],
                **{f'{pred_var}.ver': [] for pred_var in self.pred_var},
                **{f'{pred_var}.value': [] for pred_var in self.pred_var},
                **{f'{pred_var}.upper_bound': [] for pred_var in self.pred_var},
                **{f'{pred_var}.lower_bound': [] for pred_var in self.pred_var},
            }
            data_patch = {}
            data_source = {}

        source_index = data_source.get('index', [])
        for idx, (candlestick, prediction) in enumerate(zip(self.candlestick_buffer, self.prediction_buffer)):
            is_init = False
            candlestick: CandlestickBuffer
            update_idx = idx + 0.5

            if candlestick.ver:
                is_init = True

            for pred_idx, pred_var in enumerate(self.pred_var):
                if prediction[pred_idx].ver:
                    is_init = True

            if not is_init:
                continue

            if not candlestick.ts_start:
                candlestick.ts_start = self.indices[idx]

            # patching the data source
            if update_idx in source_index:
                _patch_index = source_index.index(update_idx)
                _source_candlestick_ver = _ver_list[idx] if (_ver_list := data_source.get('candlestick.ver', [])).__len__() > idx else -1

                # patching candlestick
                if _source_candlestick_ver != candlestick.ver:
                    self._patch_candlestick(patch_index=_patch_index, data_patch=data_patch, candlestick=candlestick)

                # patching prediction
                for pred_idx, pred_var in enumerate(self.pred_var):
                    prediction_buffer = prediction[pred_idx]
                    _prediction_ver = _ver_list[idx] if (_ver_list := data_source.get(f'{pred_var}.ver', [])).__len__() > idx else -1
                    if _prediction_ver != prediction_buffer.ver:
                        self._patch_prediction(patch_index=_patch_index, data_patch=data_patch, prediction_buffer=prediction_buffer, pred_var=pred_var)

                continue

            # piping the data source
            data_pipe['index'].append(update_idx)
            self._pipe_candlestick(data_pipe=data_pipe, candlestick=candlestick)
            self._pipe_prediction(data_pipe=data_pipe, prediction=prediction)

        return data_pipe, data_patch, data_source

    def _pipe_candlestick(self, data_pipe: dict[str, list], candlestick: CandlestickBuffer):
        data_pipe['market_time'].append(datetime.datetime.fromtimestamp(candlestick.ts_start, tz=self.profile.time_zone))
        data_pipe['candlestick.ver'].append(candlestick.ver)

        if not candlestick.ver:
            data_pipe['candlestick.open_price'].append(np.nan)
            data_pipe['candlestick.close_price'].append(np.nan)
            data_pipe['candlestick.high_price'].append(np.nan)
            data_pipe['candlestick.low_price'].append(np.nan)
            data_pipe['candlestick.volume'].append(np.nan)
            data_pipe['candlestick.upper_bound'].append(np.nan)
            data_pipe['candlestick.lower_bound'].append(np.nan)
            data_pipe['candlestick.color'].append('black')
        else:
            data_pipe['candlestick.open_price'].append(candlestick.open_price)
            data_pipe['candlestick.close_price'].append(candlestick.close_price)
            data_pipe['candlestick.high_price'].append(candlestick.high_price)
            data_pipe['candlestick.low_price'].append(candlestick.low_price)
            data_pipe['candlestick.volume'].append(candlestick.volume)
            data_pipe['candlestick.upper_bound'].append(max(candlestick.open_price, candlestick.close_price))
            data_pipe['candlestick.lower_bound'].append(min(candlestick.open_price, candlestick.close_price))
            data_pipe['candlestick.color'].append(self.theme.stick_style(candlestick.close_price - candlestick.open_price)['stick_color'])

    def _pipe_prediction(self, data_pipe: dict[str, list], prediction: Iterable[PredictionBuffer]):
        for pred_idx, prediction_buffer in enumerate(prediction):
            pred_var = self.pred_var[pred_idx]

            data_pipe[f'{pred_var}.ver'].append(prediction_buffer.ver)

            if not prediction_buffer.value:
                data_pipe[f'{pred_var}.value'].append(self.last_pred[pred_var].get(f'{pred_var}.value', np.nan))
                data_pipe[f'{pred_var}.upper_bound'].append(self.last_pred[pred_var].get(f'{pred_var}.upper_bound', np.nan))
                data_pipe[f'{pred_var}.lower_bound'].append(self.last_pred[pred_var].get(f'{pred_var}.lower_bound', np.nan))
            else:
                data_pipe[f'{pred_var}.value'].append(prediction_buffer.value)
                data_pipe[f'{pred_var}.upper_bound'].append(prediction_buffer.upper_bound)
                data_pipe[f'{pred_var}.lower_bound'].append(prediction_buffer.lower_bound)

    def _patch_candlestick(self, patch_index: int, data_patch: dict[str, list], candlestick: CandlestickBuffer):
        data_patch['candlestick.ver'].append((patch_index, candlestick.ver))
        data_patch['candlestick.open_price'].append((patch_index, candlestick.open_price))
        data_patch['candlestick.high_price'].append((patch_index, candlestick.high_price))
        data_patch['candlestick.low_price'].append((patch_index, candlestick.low_price))
        data_patch['candlestick.close_price'].append((patch_index, candlestick.close_price))
        data_patch['candlestick.volume'].append((patch_index, candlestick.volume))
        data_patch['candlestick.upper_bound'].append((patch_index, max(candlestick.open_price, candlestick.close_price)))
        data_patch['candlestick.lower_bound'].append((patch_index, min(candlestick.open_price, candlestick.close_price)))
        data_patch['candlestick.color'].append((patch_index, self.theme.stick_style(candlestick.close_price - candlestick.open_price)['stick_color']))

    def _patch_prediction(self, patch_index: int, data_patch: dict[str, list], prediction_buffer: PredictionBuffer, pred_var: str):
        data_patch[f'{pred_var}.ver'].append((patch_index, prediction_buffer.ver))
        data_patch[f'{pred_var}.value'].append((patch_index, prediction_buffer.value))
        data_patch[f'{pred_var}.upper_bound'].append((patch_index, prediction_buffer.upper_bound))
        data_patch[f'{pred_var}.lower_bound'].append((patch_index, prediction_buffer.lower_bound))

    def layout(self, doc_id: int):
        doc = self.bokeh_documents[doc_id]
        source = self.bokeh_data_source[doc_id]
        configs = {
            'color': {pred_var: glasbey_light[_] for _, pred_var in enumerate(self.pred_var)},
            'crosshair.x': Span(dimension="height", line_dash="dashed", line_width=1)
        }

        candlestick_plot = self._layout_candlestick(source=source, configs=configs)
        candlestick_plot.min_height = 320
        candlestick_plot.min_border_top = 0
        configs['x_axis'] = x_range = candlestick_plot.x_range
        configs['y_axis'] = y_range = candlestick_plot.y_range

        range_selector = self._layout_ranger(source=source, configs=configs)
        range_selector.min_height = 80
        range_selector.min_border_bottom = 0

        pred_plots = []
        for pred_idx, pred_var in enumerate(self.pred_var):
            _pred_plot = self._layout_prediction(pred_var=pred_var, source=source, configs=configs)
            _pred_plot.min_height = 160
            pred_plots.append(_pred_plot)

        root = gridplot(
            children=[
                [range_selector],
                [candlestick_plot],
                *[[_pred_plot] for _pred_plot in pred_plots],
            ],
            sizing_mode="stretch_both",
            merge_tools=True,
            toolbar_location='right'
        )
        # root.rows = [
        #     '40%' if self.pred_var else '80%',
        #     '10%' if self.pred_var else '20%',
        #     *[f'{50 / len(self.pred_var):.2}%' for _ in self.pred_var]
        # ]
        root.rows = [
            '1fr',
            '4fr',
            *[f'2fr' for _ in self.pred_var]
        ]
        root.min_height = sum([min_height if (min_height := _[0].min_height) else 20 for _ in root.children])
        root.width_policy = 'max'
        root.height_policy = 'max'
        root.toolbar.autohide = True
        # root.syncable = False
        # root.outline_line_color = None

        doc.add_root(root)

    def _layout_ranger(self, source: ColumnDataSource, configs: dict[str, ...]) -> figure:
        tooltips = [
            ("market_time", f"@market_time{self.theme.time_format}"),
            ("market_price", f"@{{candlestick.close_price}}{self.theme.num_format}"),
        ]

        tools = [
            HoverTool(mode='vline', syncable=False, formatters={'@market_time': 'datetime'}),
            CrosshairTool(
                overlay=[
                    Span(dimension="width", line_dash="dashed", line_width=1),
                    configs['crosshair.x']
                ]
            )
        ]

        range_selector = figure(
            title=f"Market: {self.ticker}",
            y_range=configs['y_axis'],
            # min_height=20,
            # tools=tools,
            # tooltips=tooltips,
            toolbar_location=None,
            # sizing_mode="stretch_both"
        )

        range_tool = RangeTool(x_range=configs['x_axis'])
        range_tool.overlay.fill_alpha = 0.5

        range_selector.line(
            x='index',
            y='candlestick.close_price',
            source=source
        )
        range_selector.syncable = False
        range_selector.add_tools(range_tool)
        range_selector.x_range.range_padding = self.theme.range_padding
        range_selector.xaxis.visible = False
        range_selector.yaxis.visible = False
        range_selector.xgrid.visible = False
        range_selector.ygrid.visible = False
        # range_selector.x_range.start = 0  # Set start to min value of x
        # range_selector.x_range.end = max(self.indices)  # Set end to max value of x

        # Remove the space (margin) at the ends of the main plot
        # range_selector.min_border_left = 0
        # range_selector.min_border_right = 0
        # range_selector.min_border_top = 0
        # range_selector.min_border_bottom = 0

        # range_selector.yaxis.major_label_text_font_size = '0pt'
        # range_selector.yaxis.major_tick_line_color = None
        # range_selector.yaxis.minor_tick_line_color = None

        return range_selector

    def _layout_candlestick(self, source: ColumnDataSource, configs: dict[str, ...]) -> figure:
        tooltips = [
            ("market_time", f"@market_time{self.theme.time_format}"),
            ("close_price", f"@{{candlestick.close_price}}{self.theme.num_format}"),
            ("open_price", f"@{{candlestick.open_price}}{self.theme.num_format}"),
            ("high_price", f"@{{candlestick.high_price}}{self.theme.num_format}"),
            ("low_price", f"@{{candlestick.low_price}}{self.theme.num_format}")
        ]

        tools = [
            PanTool(dimensions="width", syncable=False),
            WheelPanTool(dimension="width", syncable=False),
            BoxZoomTool(dimensions="auto", syncable=False),
            WheelZoomTool(dimensions="width", syncable=False),
            CrosshairTool(
                overlay=[
                    Span(dimension="width", line_dash="dashed", line_width=1),
                    configs['crosshair.x']
                ]
            ),
            HoverTool(mode='vline', syncable=False, formatters={'@market_time': 'datetime'}),
            ExamineTool(syncable=False),
            ResetTool(syncable=False),
            SaveTool(syncable=False)
        ]

        plot = figure(
            x_range=Range1d(start=0, end=len(self.indices), bounds='auto'),
            x_axis_type="linear",
            # sizing_mode="stretch_both",
            # min_height=80,
            tools=tools,
            tooltips=tooltips,
            y_axis_location="right",
        )

        _shadows = plot.segment(
            name='candlestick.shade',
            x0='index',
            x1='index',
            y0='candlestick.low_price',
            y1='candlestick.high_price',
            line_width=1,
            color="black",
            alpha=0.8,
            source=source
        )

        _candlestick = plot.vbar(
            name='candlestick',
            x='index',
            top='candlestick.upper_bound',
            bottom='candlestick.lower_bound',
            width=1 - self.theme.stick_padding,
            color='candlestick.color',
            alpha=0.5,
            source=source
        )

        plot.xaxis.major_label_overrides = {i: datetime.datetime.fromtimestamp(ts, tz=self.profile.time_zone).strftime('%Y-%m-%d\n%H:%M:%S') for i, ts in enumerate(self.indices)}
        plot.xaxis.ticker = SingleIntervalTicker(interval=(15 * 60 / self.interval), num_minor_ticks=3)
        # plot.xaxis.ticker.min_interval = 1.
        tools[5].renderers = [_candlestick]
        return plot

    def _layout_prediction(self, pred_var: str, source: ColumnDataSource, configs: dict[str, ...]):
        tooltips = [
            ("market_time", f"@market_time{self.theme.time_format}"),
            (f"{pred_var}.value", f"@{{{pred_var}.value}}{self.theme.pct_format}"),
            (f"{pred_var}.upper_bound", f"@{{{pred_var}.upper_bound}}{self.theme.pct_format}"),
            (f"{pred_var}.lower_bound", f"@{{{pred_var}.lower_bound}}{self.theme.pct_format}")
        ]

        tools = [
            HoverTool(mode='vline', syncable=False, formatters={'@market_time': 'datetime'}),
            CrosshairTool(
                overlay=[
                    Span(dimension="width", line_dash="dashed", line_width=1),
                    configs['crosshair.x']
                ]
            )
        ]

        # Prediction Subplot
        plot = figure(
            title=f"Prediction: {pred_var}",
            x_range=configs['x_axis'],  # Share x-axis range
            # min_height=40,
            tools=tools,
            tooltips=tooltips,
            toolbar_location=None
        )

        # Vertical line y = 0
        # zero_line = Span(location=0, dimension='width', line_color='black', line_dash='dashed', line_width=1)
        # plot.add_layout(zero_line)

        # Shaded area between upper_bound and lower_bound (50% transparency)
        plot.varea(
            x='index',
            y1=f'{pred_var}.upper_bound',
            y2=f'{pred_var}.lower_bound',
            source=source,
            fill_color=configs['color'][pred_var],
            fill_alpha=0.2,
            # hatch_pattern='x'
        )

        # Line plot for predicted state
        _pred_line = plot.line(
            x='index',
            y=f'{pred_var}.value',
            source=source,
            line_width=2,
            color=configs['color'][pred_var]
        )

        # Dashed borderlines for upper_bound and lower_bound
        plot.line(
            x='index',
            y=f'{pred_var}.upper_bound',
            source=source,
            line_width=1,
            color=configs['color'][pred_var],
            line_dash="dashed"  # Dashed line for upper bound
        )

        plot.line(
            x='index',
            y=f'{pred_var}.lower_bound',
            source=source,
            line_width=1,
            color=configs['color'][pred_var],
            line_dash="dashed"  # Dashed line for lower bound
        )

        hover_tool = tools[0]
        if isinstance(hover_tool.renderers, list):
            hover_tool.renderers.append(_pred_line)
        else:
            hover_tool.renderers = [_pred_line]

        plot.xaxis.fixed_location = 0
        plot.xaxis.ticker = FixedTicker(ticks=[])
        # plot.yaxis.formatter = TickFormatter(format='0.0000%')

        match pred_var.split('_'):
            case ['state', *_] | ['markov', *_]:
                plot.y_range.bounds = (-1, 1)
                plot.yaxis.ticker = FixedTicker(ticks=[-1, -0.5, 0, 0.5, 1])
            case ['up', *_]:
                plot.y_range.bounds = (0, None)
                plot.yaxis.ticker = SingleIntervalTicker(interval=0.001, num_minor_ticks=2)
            case ['down', *_]:
                plot.y_range.bounds = (None, 0)
                plot.yaxis.ticker = SingleIntervalTicker(interval=0.001, num_minor_ticks=2)
            case _:
                plot.yaxis.ticker = SingleIntervalTicker(interval=0.001, num_minor_ticks=2)

        plot.yaxis[0].formatter = NumeralTickFormatter(format='0.00%')
        return plot

    @property
    def data(self):
        data_pipe, data_patch, data_source = self._update_datasource()
        return data_pipe
