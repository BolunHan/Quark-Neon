import abc
import json
import os.path
import pathlib
import re
from typing import Literal

from algo_engine.engine import PositionManagementService, MarketDataService

from . import LOGGER
from ..base import CONFIG


class NameSpace(dict):
    def __init__(self):
        super().__init__()
        self.__dict__ = self


class StrategyProfile(NameSpace):
    """
    a dict to store env variables of the strategy
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.sampling_interval: float = kwargs.get('sampling_interval', CONFIG.Statistics.FACTOR_SAMPLING_INTERVAL)
        self.clear_on_eod: bool = kwargs.get('clear_on_eod', True)


class StrategyState(NameSpace):
    """
    a dictionary to store the state of strategy
    """

    def __init__(self, **kwargs):
        super().__init__()

        # market status
        self.timestamp: float = kwargs.get('timestamp', 0.)

        # balance status
        self.ticker: str | None = kwargs.get('ticker')
        self.exposure: float = kwargs.get('exposure', 0.)
        self.working_long: float = kwargs.get('working_long', 0.)
        self.working_short: float = kwargs.get('working_short', 0.)

        self.pos_pnl: float = kwargs.get('pos_pnl', 0.)  # pnl for current position
        self.pos_mdd: float = kwargs.get('pos_mdd', 0.)  # max drawdown for current position
        self.pos_ts: float = kwargs.get('pos_ts', 0.)  # holding time (in seconds) for current position
        self.pos_cash_flow = kwargs.get('pos_cash_flow', 0.)
        self.pos_start_ts = kwargs.get('pos_start_ts', 0.)

    def update_market_state(self, mds: MarketDataService, **kwargs):
        self.timestamp = mds.timestamp

    def update_balance_state(self, position: PositionManagementService, ticker: str, **kwargs):
        if position is None:
            LOGGER.warning('position not given, assuming no position. NOTE: Only gives empty position in BACKTEST mode!')
            exposure_volume = working_volume = 0
        else:
            exposure_volume = position.exposure_volume
            working_volume = position.working_volume

        self.timestamp = position.dma.mds.timestamp
        self.ticker = ticker
        self.exposure = exposure_volume.get(ticker, 0.)
        self.working_long = working_volume['Long'].get(ticker, 0.)
        self.working_short = working_volume['Short'].get(ticker, 0.)

        if self.exposure:
            # self.pos_pnl = position.market_price[ticker] * self.exposure + self.pos_cash_flow
            # pos_cash_flow and pos_start_ts should be updated by Strategy
            self.pos_mdd = min(self.pos_pnl, self.pos_mdd)
            self.pos_ts = self.timestamp - self.pos_start_ts
        else:
            self.pos_pnl = 0.
            self.pos_cash_flow = 0.
            self.pos_start_ts = 0.
            self.pos_mdd = 0.
            self.pos_ts = 0.


class DecisionCore(object, metaclass=abc.ABCMeta):
    def __init__(self, profile: StrategyProfile = None, state: StrategyState = None):
        self.profile: StrategyProfile = StrategyProfile() if profile is None else profile
        self.state: StrategyState = StrategyState() if state is None else state

    @abc.abstractmethod
    def predict(self, mds: MarketDataService, **kwargs) -> dict[str, float]:
        ...

    def predict_full(self, mds: MarketDataService, **kwargs) -> dict[str, float]:
        """
        return full prediction value, not the selected ones.
        """
        return self.predict(mds=mds, **kwargs)

    @abc.abstractmethod
    def signal(self, mds: MarketDataService, position: PositionManagementService, **kwargs) -> Literal[-1, 0, 1]:
        ...

    @abc.abstractmethod
    def trade_volume(self, position: PositionManagementService, cash: float, margin: float, timestamp: float, signal: int) -> float:
        ...

    @abc.abstractmethod
    def calibrate(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def validation(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def clear(self):
        ...

    @abc.abstractmethod
    def to_json(self, fmt='dict') -> dict | str:
        ...

    @classmethod
    @abc.abstractmethod
    def from_json(cls, json_str: str | bytes | dict):
        ...

    def dump(self, file_path: str | pathlib.Path = None):
        json_dict = self.to_json(fmt='dict')
        json_str = json.dumps(json_dict, indent=4)

        if file_path is not None:
            with open(file_path, 'w') as f:
                f.write(json_str)

        return json_dict

    @classmethod
    def load(cls, file_path: str | pathlib.Path = None, file_pattern: str | re.Pattern = None, file_dir: str | pathlib.Path = None):
        if file_path is not None:
            if os.path.isfile(file_path):
                with open(file_path, 'r') as f:
                    json_dict = json.load(f)
                return cls.from_json(json_dict)
            else:
                raise FileNotFoundError(f'{file_path} does not exist!')
        elif file_pattern is not None:
            if not os.path.isdir(file_dir):
                raise FileNotFoundError(f'{file_dir} does not exist!')
            else:
                for file_name in sorted(os.listdir(file_dir), reverse=True):
                    if re.match(file_pattern, file_name):
                        if file_dir is None:
                            file_path = os.path.realpath(file_name)
                        else:
                            file_path = os.path.realpath(pathlib.Path(file_dir, file_name))

                        LOGGER.info(f'loading {cls.__name__} from {file_path}...')

                        return cls.load(file_path=file_path)

                if file_dir is None:
                    raise FileNotFoundError(f'{file_pattern} does not exist!')
                else:
                    raise FileNotFoundError(f'{file_pattern} does not exist at {os.path.realpath(file_dir)}!')
        else:
            raise ValueError('Must assign a file_path or a file_pattern')

    @property
    @abc.abstractmethod
    def is_ready(self):
        ...


class DummyDecisionCore(DecisionCore):

    def predict(self, mds: MarketDataService, **kwargs) -> dict[str, float]:
        return {}

    def signal(self, mds: MarketDataService, position: PositionManagementService, **kwargs) -> int:
        return 0

    def trade_volume(self, position: PositionManagementService, cash: float, margin: float, timestamp: float, signal: int) -> float:
        return 1.

    def calibrate(self, *args, **kwargs) -> dict:
        """
        calibrate decision core and gives a calibration report, in dict (like json)
        """
        pass

    def validation(self, *args, **kwargs):
        pass

    def clear(self):
        pass

    def to_json(self, fmt='dict') -> dict | str:
        pass

    @classmethod
    def from_json(cls, json_str: str | bytes | dict):
        pass

    @property
    def is_ready(self):
        return True
