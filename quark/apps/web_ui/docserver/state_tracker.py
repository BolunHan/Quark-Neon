import ctypes
from functools import partial
from typing import TypedDict, Literal

import numpy as np
from algo_engine.apps import DocServer, DocTheme
from algo_engine.base import MarketData
from algo_engine.profile import Profile, PROFILE
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, Div


class StateBuffer(ctypes.Structure):
    _fields_ = [
        ('ver', ctypes.c_int),
        ('position', ctypes.c_int),
        ('signal', ctypes.c_double),
        ('n_action', ctypes.c_int),
        ('net_value', ctypes.c_double),
        ('ttl_fee', ctypes.c_double),
        ('ttl_pnl', ctypes.c_double),
        ('cash_flow', ctypes.c_double)
    ]


class StateBannerTheme(DocTheme):
    stick_padding = 0.1
    range_padding = 0.01

    ColorStyle = TypedDict('ColorStyle', fields={'up': str, 'down': str})
    ws_style = ColorStyle(up="green", down="red")
    cn_style = ColorStyle(up="red", down="green")
    num_format = '{0,0.00}'
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


class StateBanner(DocServer):
    def __init__(self, ticker: str, profile: Profile = PROFILE, theme: DocTheme = None, **kwargs):
        self.ticker = ticker
        self.profile = profile
        self.fee_rate = kwargs.get('fee_rate', 0.00016)

        super().__init__(
            theme=theme,
            max_size=kwargs.get('max_size'),
            update_interval=kwargs.get('update_interval', 0),
        )

        self.theme = StateBannerTheme(profile=self.profile) if self.theme is None else self.theme
        self.state_buffer = StateBuffer()
        self.market_price = np.nan
        self.timestamp: float = 0.

    def update(self, **kwargs):
        kwargs = kwargs.copy()
        is_updated = False

        # step 1: update candlesticks
        if 'market_data' in kwargs:
            market_data: MarketData = kwargs['market_data']

            if market_data.ticker != self.ticker:
                return

            self.market_price = market_data.market_price
            self.timestamp = market_data.timestamp
            self.state_buffer.net_value = self.state_buffer.position * self.market_price + self.state_buffer.cash_flow
            self.state_buffer.ver += 1
            is_updated = True
        elif 'ticker' in kwargs and 'market_price' in kwargs:
            ticker = kwargs['ticker']
            timestamp = kwargs.get('timestamp', self.timestamp)
            market_price = kwargs.get('close_price', kwargs.get('market_price'))

            if ticker != self.ticker:
                return

            self.market_price = market_price
            self.timestamp = timestamp
            self.state_buffer.net_value = self.state_buffer.position * self.market_price + self.state_buffer.cash_flow
            self.state_buffer.ver += 1
            is_updated = True

        if 'signal' in kwargs:
            signal = kwargs['signal']

            if np.isfinite(signal):
                self.state_buffer.signal = kwargs['signal']
                self.state_buffer.ver += 1
                is_updated = True

        if 'action' in kwargs:
            action = kwargs['action']

            if not action and not is_updated:
                return

            current_position = self.state_buffer.position

            # if current_position * action > 0 and not is_updated:
            #     return

            cash_flow = -action * self.market_price
            target_position = current_position + action

            self.state_buffer.cash_flow += cash_flow
            self.state_buffer.position = target_position
            self.state_buffer.n_action += 1 if action else 0
            self.state_buffer.ttl_fee += abs(cash_flow) * self.fee_rate

            if target_position == 0:
                self.state_buffer.ttl_pnl += self.state_buffer.cash_flow
                self.state_buffer.cash_flow = 0

            self.state_buffer.net_value = self.state_buffer.position * self.market_price + self.state_buffer.cash_flow
            self.state_buffer.ver += 1
            is_updated = True

        if is_updated:
            for doc_id in list(self.bokeh_documents):
                doc = self.bokeh_documents[doc_id]
                self._update_datasource(doc_id=doc_id)

                if not self.update_interval:
                    doc.add_next_tick_callback(partial(self.stream, doc_id=doc_id))
                    doc.add_next_tick_callback(partial(self.patch, doc_id=doc_id))

    def _update_banner(self, attr, old, new, div: Div, topic: str, title: str = None, formatter: str = None, side: Literal['up', 'down', 'neutral', 'none'] = None, comments: str = ''):
        data = new[topic]
        value = data[0] if data else np.nan

        if side is None:
            side = 'up' if value > 0 else 'down' if value < 0 else 'neutral'

        # Determine the color and HTML entity for advanced arrows
        if side == 'up':
            color = self.theme.style['up']
            icon_up = '&#11205;'  # Up arrow
            icon_down = ''
        elif side == 'down':
            color = self.theme.style['down']
            icon_up = ''
            icon_down = '&#11206;'  # Down arrow
        elif side == 'neutral':
            color = 'none'
            icon_up = '&#11205;'
            icon_down = '&#11206;'
        elif side == 'none':
            color = 'none'
            icon_up = ''
            icon_down = ''
        else:
            raise ValueError(f'Invalid side {side}!')

        if title is None:
            title = f'Current {topic.capitalize()}'

        if formatter is None:
            contexts = f'{value}'
        else:
            contexts = formatter.format(value)

        # Create the HTML structure with the new class name pattern
        text = f"""
        <style>
            .bk-clearfix {{
                display: flex !important;
                width: 100% !important;
            }}
            .banner_state_{topic} {{
                display: flex;
                flex-direction: column;
                height: 100%;
                width: 100% !important; /* Full width */
                justify-content: space-between;  /* Distribute space evenly */
                filter: drop-shadow(8px 4px 4px rgba(0, 0, 0, 0.2));  /* Light shadow for icon */
            }}
            .banner_state_{topic}_top {{
                height: 2fr;
                text-align: center;
                font-weight: bold;
            }}
            .banner_state_{topic}_middle {{
                height: 4fr;
                width: 100%;
                display: flex;
            }}
            .banner_state_{topic}_icon {{
                width: 10%;  /* 20% width for the icon */
                display: flex;
                justify-content: flex-start;  /* Center the icon */
                font-size: 1.5em;  /* Size of the icon */
                color: {color};
            }}
            .banner_state_{topic}_text {{
                width: 80%;  /* 80% width for the text */
                text-align: right;  /* Right align the text */
                font-size: 1.5em;  /* Size of the text */
                color: {color};
            }}
            .banner_state_{topic}_bottom {{
                height: {"1fr" if comments else "0fr"};
                text-align: right;
                font-style: italic;
                color: #6c757d;
                font-weight: lighter;
            }}
            .banner_state_{topic}_hr {{
                width: 80%;
                border: 0;
                /* border-top: 1px solid #ccc; */
                margin: 0 auto;  /* Center the horizontal line */
            }}
        </style>
        <div class="banner_state_{topic}">
            <div class="banner_state_{topic}_top">
                <h3>{title}</h3>  <!-- Title Text -->
            </div>
            <hr class="banner_state_{topic}_hr">  <!-- Horizontal Line -->
            <div class="banner_state_{topic}_middle">
                <div class="banner_state_{topic}_icon">
                    {icon_up} 
                </div>
                <div class="banner_state_{topic}_icon">
                    {icon_down}  
                </div>
                <div class="banner_state_{topic}_text">
                    {contexts}  <!-- Value with HTML entity arrow -->
                </div>
            </div>
            <hr class="banner_state_{topic}_hr">  <!-- Horizontal Line -->
            {f"""<div class="banner_state_{topic}_bottom">
                <span>{comments}</span>
            </div>""" if comments else ""}
        </div>
        """

        div.text = text
        return text

    def _update_datasource(self, doc_id: int = None) -> tuple[dict[str, list], dict[str, list[tuple]], dict[str, list]]:
        if doc_id is not None:
            data_pipe = self.bokeh_data_pipe[doc_id]
            data_patch = self.bokeh_data_patch[doc_id]
            data_source = self.bokeh_data_source[doc_id].data
        else:
            data_pipe = {
                'ver': [],  # the plotting index
                'position': [],
                'signal': [],
                'n_action': [],
                'net_value': [],
                'ttl_fee': [],
                'ttl_pnl': [],
                'cash_flow': []
            }
            data_patch = {}
            data_source = {}

        if not self.state_buffer.ver:
            pass
        elif not data_source.get('ver'):
            data_pipe['ver'].append(self.state_buffer.ver)
            data_pipe['position'].append(self.state_buffer.position)
            data_pipe['signal'].append(self.state_buffer.signal)
            data_pipe['n_action'].append(self.state_buffer.n_action)
            data_pipe['net_value'].append(self.state_buffer.net_value)
            data_pipe['ttl_fee'].append(self.state_buffer.ttl_fee)
            data_pipe['ttl_pnl'].append(self.state_buffer.ttl_pnl)
            data_pipe['cash_flow'].append(self.state_buffer.cash_flow)
        elif data_source['ver'] != self.state_buffer.ver:
            data_patch['ver'].append((0, self.state_buffer.ver))
            data_patch['position'].append((0, self.state_buffer.position))
            data_patch['signal'].append((0, self.state_buffer.signal))
            data_patch['n_action'].append((0, self.state_buffer.n_action))
            data_patch['net_value'].append((0, self.state_buffer.net_value))
            data_patch['ttl_fee'].append((0, self.state_buffer.ttl_fee))
            data_patch['ttl_pnl'].append((0, self.state_buffer.ttl_pnl))
            data_patch['cash_flow'].append((0, self.state_buffer.cash_flow))
        else:
            pass

        return data_pipe, data_patch, data_source

    def layout(self, doc_id: int):
        doc = self.bokeh_documents[doc_id]
        source = self.bokeh_data_source[doc_id]
        configs = {}

        position_div = self._layout_div(source=source, configs=configs)
        self._update_banner(attr='data', old=source.data, new=source.data, div=position_div, topic='position')
        source.on_change('data', partial(self._update_banner, div=position_div, topic='position'))

        signal_div = self._layout_div(source=source, configs=configs)
        self._update_banner(attr='data', old=source.data, new=source.data, div=signal_div, topic='signal', formatter='{:.2%}')
        source.on_change('data', partial(self._update_banner, div=signal_div, topic='signal', formatter='{:.2%}'))

        action_div = self._layout_div(source=source, configs=configs)
        self._update_banner(attr='data', old=source.data, new=source.data, div=action_div, topic='n_action', title='Num of Actions', formatter='{:,}', side='none')
        source.on_change('data', partial(self._update_banner, div=action_div, topic='n_action', title='Num of Actions', formatter='{:,}', side='none'))

        fee_div = self._layout_div(source=source, configs=configs)
        self._update_banner(attr='data', old=source.data, new=source.data, div=fee_div, topic='ttl_fee', title='Est. Fee', formatter='-{:.4f}', side='down', comments=f'--by Fee Rate {self.fee_rate:.4%}')
        source.on_change('data', partial(self._update_banner, div=fee_div, topic='ttl_fee', title='Est. Fee', formatter='-{:.4f}', side='down', comments=f'--  by fee rate {self.fee_rate:.4%}'))

        pnl_div = self._layout_div(source=source, configs=configs)
        self._update_banner(attr='data', old=source.data, new=source.data, div=pnl_div, topic='ttl_pnl', title='Est. PnL', formatter='{:.2f}')
        source.on_change('data', partial(self._update_banner, div=pnl_div, topic='ttl_pnl', title='Est. PnL', formatter='{:.2f}'))

        nav_div = self._layout_div(source=source, configs=configs)
        self._update_banner(attr='data', old=source.data, new=source.data, div=nav_div, topic='net_value', title='Est. NAV', formatter='{:.2f}', comments='--NAV for Current position')
        source.on_change('data', partial(self._update_banner, div=nav_div, topic='net_value', title='Est. NAV', formatter='{:.2f}', comments='--  NAV for Current position'))

        root = row(
            children=[
                position_div,
                self.create_vl_div(),
                signal_div,
                self.create_vl_div(),
                action_div,
                self.create_vl_div(),
                fee_div,
                self.create_vl_div(),
                pnl_div,
                self.create_vl_div(),
                nav_div
            ],
            sizing_mode='stretch_width',  # Make the row stretch to fit the available space
            margin=(0, 30, 0, 30),  # Set margins to zero
            spacing=10  # Set spacing to zero
        )

        doc.add_root(root)

    def _layout_div(self, source: ColumnDataSource, configs: dict[str, ...]) -> Div:
        tooltips = [
            ("position", f"@position{self.theme.num_format}")
        ]

        position_data = source.data['position']
        div_value = position_data[0] if position_data else np.nan

        # Create a Div to display dynamic content
        div = Div(
            # text=f"""<h1 title="Current Position:\n{div_value}!">{div_value}</h1>""",
            # tooltips=tooltips
            # width=200,  # Set a fixed width or use CSS for responsive design
            styles={'flex': '1'},  # This will allow it to expand and take equal space in the row
        )

        return div

    def create_vl_div(self) -> Div:
        """
        Creates a vertical line divider (vl) between divs in the layout.
        The line is white, 90% of the height of its container, and centered vertically.
        """
        text = f"""
        <style>
            /* Ensure the parent container uses flexbox and allows full height */
            .bk-clearfix {{
                display: flex !important;
                height: 100% !important;
            }}

            .vl {{
                width: 2px;  /* Fixed width for the vertical line */
                height: 90%;  /* 90% of the parent container's height */
                opacity: 0.5;  /* Alpha value for transparency */
                background-color: currentColor;  /* Line color */
                margin: auto 0;  /* Vertically center the line */
                display: flex;  /* Ensure it aligns inside flex containers */
                justify-content: center;
                align-items: center;
            }}
        </style>
        <div class="vl"></div>  <!-- Vertical line div -->
        """

        return Div(text=text)

    @property
    def data(self):
        data_pipe, data_patch, data_source = self._update_datasource()
        return data_pipe

    @property
    def position(self) -> int:
        return self.state_buffer.position
