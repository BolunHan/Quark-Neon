import inspect
import os
import pathlib
from types import SimpleNamespace

from EventEngine import Topic, PatternTopic
from PyQuantKit import MarketData

from . import Exceptions

__all__ = ['GlobalStatics']


def get_current_path(idx: int = 1) -> pathlib.Path:
    stacks = inspect.stack()
    if len(stacks) < 1:
        raise ValueError(f'Can not go back to {idx} stack')
    else:
        return pathlib.Path(stacks[idx].filename)


FILE_PATH = get_current_path()


class _Constants(object):
    def __init__(self, value: str):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f'<GlobalConstants>({self.value})'


class _TopicSet(object):
    on_order = Topic('on_order')
    on_report = Topic('on_report')
    launch_signal = Topic('launch_signal')
    eod = Topic('eod')
    eod_done = Topic('eod_done')
    bod = Topic('bod')
    bod_done = Topic('bod_done')

    launch_order = PatternTopic('launch_order.{ticker}')
    cancel_order = PatternTopic('cancel_order.{ticker}')
    query = PatternTopic('query.{ticker}.{dtype}')
    subscribe = PatternTopic('subscribe.{ticker}.{dtype}')
    realtime = PatternTopic('realtime.{ticker}.{dtype}')
    history = PatternTopic('history.{ticker}.{dtype}')
    on_open = PatternTopic('open.{source}')
    on_close = PatternTopic('close.{source}')

    @classmethod
    def push(cls, market_data: MarketData):
        return cls.realtime(ticker=market_data.ticker, dtype=market_data.__class__.__name__)

    @classmethod
    def parse(cls, topic: Topic) -> SimpleNamespace:
        try:
            _ = topic.value.split('.')

            action = _.pop(0)
            if action in ['open', 'close']:
                dtype = None
            else:
                dtype = _.pop(-1)
            ticker = '.'.join(_)

            p = SimpleNamespace(
                action=action,
                dtype=dtype,
                ticker=ticker
            )
            return p
        except Exception as _:
            raise Exceptions.TopicError(f'Invalid topic {topic}')


class _GlobalStatics(object):
    def __init__(self):
        self.NO_UPDATE = _Constants('Signal.NO_UPDATE')
        self.NO_SIGNAL = _Constants('Signal.NO_SIGNAL')
        self.CLOSE_SIGNAL = _Constants('Signal.CLOSE_SIGNAL')
        self.CANCEL_SIGNAL = _Constants('Signal.CANCEL_SIGNAL')
        self.CURRENCY = 'CNY'
        self.SHARE = 'SHARE'
        self.TOPIC = _TopicSet

        if 'QUARK_CWD' in os.environ:
            self.WORKING_DIRECTORY = _Constants(os.path.realpath(os.environ['QUARK_CWD']))
        else:
            # self.WORKING_DIRECTORY = _Constants(str(FILE_PATH.parent.parent))
            self.WORKING_DIRECTORY = _Constants(str(os.getcwd()))

        os.makedirs(self.WORKING_DIRECTORY.value, exist_ok=True)

    def add_static(self, name: str, value):
        setattr(self, name, _Constants(value))


GlobalStatics = _GlobalStatics()
