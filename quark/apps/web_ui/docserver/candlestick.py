import ctypes
import datetime
import pathlib
from functools import partial
from typing import TypedDict

import pandas as pd
from algo_engine.apps import DocServer, DocTheme
from algo_engine.base import MarketData, TradeData, TransactionData
from algo_engine.profile import Profile, PROFILE
from algo_engine.utils import ts_indices

from . import LOGGER


class CandlestickBuffer(ctypes.Structure):
    _fields_ = [
        # ('is_ready', ctypes.c_bool),
        ('ver', ctypes.c_uint),
        ('ts_start', ctypes.c_double),
        ('ts_end', ctypes.c_double),
        ('open_price', ctypes.c_double),
        ('close_price', ctypes.c_double),
        ('high_price', ctypes.c_double),
        ('low_price', ctypes.c_double),
        ('volume', ctypes.c_double)
    ]


class StickTheme(DocTheme):
    stick_padding = 0.1
    range_padding = 0.01

    ColorStyle = TypedDict('ColorStyle', fields={'up': str, 'down': str})
    ws_style = ColorStyle(up="green", down="red")
    cn_style = ColorStyle(up="red", down="green")

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


class Candlestick(DocServer):
    def __init__(self, ticker: str, start_date: datetime.date, end_date: datetime.date, profile: Profile = PROFILE, interval: float = 60., x_axis: list[float] = None, theme: DocTheme = None, **kwargs):
        self.ticker = ticker
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

        self.theme = StickTheme(profile=self.profile) if self.theme is None else self.theme
        self.timestamp: float = 0.
        self.candlestick_buffer = (CandlestickBuffer * len(self.indices))()
        self.last_idx = 0

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
            LOGGER.warning('start timestamp greater than timestamp! Resetting the searching index!')
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

        if 'market_data' in kwargs:
            market_data: MarketData = kwargs['market_data']

            if market_data.ticker != self.ticker:
                return

            if isinstance(market_data, (TradeData, TransactionData)):
                self._update_candlestick(timestamp=market_data.timestamp, market_price=market_data.price, volume=market_data.volume)
            else:
                self._update_candlestick(timestamp=market_data.timestamp, market_price=market_data.market_price)
            self.timestamp = market_data.timestamp
        elif 'ticker' in kwargs and 'market_price' in kwargs:
            ticker = kwargs['ticker']
            timestamp = kwargs.get('timestamp', self.timestamp)
            market_price = kwargs.get('close_price', kwargs.get('market_price'))
            open_price = kwargs.get('open_price', market_price)
            high_price = kwargs.get('high_price', market_price)
            low_price = kwargs.get('low_price', market_price)
            volume = kwargs.get('volume', 0)

            assert ticker is not None, 'Must assign a ticker for update function!'
            assert market_price is not None, f'Must assign a market_price or close_price for {self.__class__} update function!'

            if ticker != self.ticker:
                return

            self._update_candlestick(
                timestamp=timestamp,
                market_price=market_price,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                volume=volume
            )
            self.timestamp = timestamp

        for doc_id in list(self.bokeh_documents):
            doc = self.bokeh_documents[doc_id]
            self._update_datasource(doc_id=doc_id)

            if not self.update_interval:
                doc.add_next_tick_callback(partial(self.stream, doc_id=doc_id))
                doc.add_next_tick_callback(partial(self.patch, doc_id=doc_id))

    def _update_candlestick(self, timestamp: float, market_price: float, open_price: float = None, high_price: float = None, low_price: float = None, volume: float = 0.):
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

    def _update_datasource(self, doc_id: int = None) -> tuple[dict[str, list], dict[str, list[tuple]], dict[str, list]]:
        if doc_id is not None:
            data_pipe = self.bokeh_data_pipe[doc_id]
            data_patch = self.bokeh_data_patch[doc_id]
            data_source = self.bokeh_data_source[doc_id].data
        else:
            data_pipe = dict(
                index=[],
                ver=[],
                market_time=[],
                open_price=[],
                high_price=[],
                low_price=[],
                close_price=[],
                volume=[],
                cs_upperbound=[],
                cs_lowerbound=[],
                cs_color=[]
            )
            data_patch = {}
            data_source = {}

        source_index = data_source.get('index', [])
        source_ver = data_source.get('ver', [])
        for idx, candlestick in enumerate(self.candlestick_buffer):
            candlestick: CandlestickBuffer
            update_idx = idx + 0.5
            update_ver = candlestick.ver

            if not update_ver:
                continue

            if update_idx in source_index:
                _patch_index = source_index.index(update_idx)
                _source_ver = source_ver[_patch_index]

                if _source_ver == update_ver:
                    continue

                data_patch['ver'].append((_patch_index, update_ver))
                data_patch['open_price'].append((_patch_index, candlestick.open_price))
                data_patch['high_price'].append((_patch_index, candlestick.high_price))
                data_patch['low_price'].append((_patch_index, candlestick.low_price))
                data_patch['close_price'].append((_patch_index, candlestick.close_price))
                data_patch['volume'].append((_patch_index, candlestick.volume))
                data_patch['cs_upperbound'].append((_patch_index, max(candlestick.open_price, candlestick.close_price)))
                data_patch['cs_lowerbound'].append((_patch_index, min(candlestick.open_price, candlestick.close_price)))
                data_patch['cs_color'].append((_patch_index, self.theme.stick_style(candlestick.close_price - candlestick.open_price)['stick_color']))
                continue

            data_pipe['ver'].append(update_ver)  # to ensure bar rendered in the center of the interval
            data_pipe['index'].append(update_idx)  # to ensure bar rendered in the center of the interval
            data_pipe['market_time'].append(datetime.datetime.fromtimestamp(candlestick.ts_start, tz=self.profile.time_zone))
            data_pipe['open_price'].append(candlestick.open_price)
            data_pipe['close_price'].append(candlestick.close_price)
            data_pipe['high_price'].append(candlestick.high_price)
            data_pipe['low_price'].append(candlestick.low_price)
            data_pipe['volume'].append(candlestick.volume)
            data_pipe['cs_upperbound'].append(max(candlestick.open_price, candlestick.close_price))
            data_pipe['cs_lowerbound'].append(min(candlestick.open_price, candlestick.close_price))
            data_pipe['cs_color'].append(self.theme.stick_style(candlestick.close_price - candlestick.open_price)['stick_color'])

        return data_pipe, data_patch, data_source

    def layout(self, doc_id: int):
        self._register_candlestick(doc_id=doc_id)

    def _register_candlestick(self, doc_id: int):
        from bokeh.models import PanTool, WheelPanTool, WheelZoomTool, BoxZoomTool, ResetTool, ExamineTool, SaveTool, CrosshairTool, HoverTool, RangeTool, Range1d
        from bokeh.plotting import figure, gridplot

        doc = self.bokeh_documents[doc_id]
        source = self.bokeh_data_source[doc_id]

        tools = [
            PanTool(dimensions="width", syncable=False),
            WheelPanTool(dimension="width", syncable=False),
            BoxZoomTool(dimensions="auto", syncable=False),
            WheelZoomTool(dimensions="width", syncable=False),
            CrosshairTool(dimensions="both", syncable=False),
            HoverTool(mode='vline', syncable=False, formatters={'@market_time': 'datetime'}),
            ExamineTool(syncable=False),
            ResetTool(syncable=False),
            SaveTool(syncable=False)
        ]

        tooltips = [
            ("market_time", "@market_time{%H:%M:%S}"),
            ("close_price", "@close_price"),
            ("open_price", "@open_price"),
            ("high_price", "@high_price"),
            ("low_price", "@low_price"),
        ]

        plot = figure(
            title=f"{self.ticker} Candlestick",
            x_range=Range1d(start=0, end=len(self.indices), bounds='auto'),
            x_axis_type="linear",
            # sizing_mode="stretch_both",
            min_height=80,
            tools=tools,
            tooltips=tooltips,
            y_axis_location="right",
        )

        _shadows = plot.segment(
            name='candlestick.shade',
            x0='index',
            x1='index',
            y0='low_price',
            y1='high_price',
            line_width=1,
            color="black",
            alpha=0.8,
            source=source
        )

        _candlestick = plot.vbar(
            name='candlestick',
            x='index',
            top='cs_upperbound',
            bottom='cs_lowerbound',
            width=1 - self.theme.stick_padding,
            color='cs_color',
            alpha=0.5,
            source=source
        )

        plot.xaxis.major_label_overrides = {i: datetime.datetime.fromtimestamp(ts, tz=self.profile.time_zone).strftime('%Y-%m-%d %H:%M:%S') for i, ts in enumerate(self.indices)}
        plot.xaxis.ticker.min_interval = 1.
        tools[5].renderers = [_candlestick]

        range_selector = figure(
            y_range=plot.y_range,
            min_height=20,
            tools=[],
            toolbar_location=None,
            # sizing_mode="stretch_both"
        )

        range_tool = RangeTool(x_range=plot.x_range)
        range_tool.overlay.fill_alpha = 0.5

        range_selector.line('index', 'close_price', source=source)
        range_selector.add_tools(range_tool)
        range_selector.x_range.range_padding = self.theme.range_padding
        range_selector.xaxis.visible = False
        range_selector.xgrid.visible = False
        range_selector.ygrid.visible = False

        root = gridplot(
            children=[
                [plot],
                [range_selector]
            ],
            sizing_mode="stretch_both",
            merge_tools=True,
            toolbar_options={
                'autohide': True,
                'active_drag': tools[0],
                'active_scroll': tools[3]
            },
        )
        root.rows = ['80%', '20%']
        root.width_policy = 'max'
        root.height_policy = 'max'

        doc.add_root(root)

    def to_csv(self, filename: str | pathlib.Path):
        df = pd.DataFrame(self.data).set_index(keys='market_time')
        df = df[['open_price', 'high_price', 'low_price', 'close_price', 'volume']]
        df.to_csv(filename)

    @property
    def data(self):
        data_pipe, data_patch, data_source = self._update_datasource()
        return data_pipe
