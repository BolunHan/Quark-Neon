import abc
import datetime
import enum
import inspect
import json
import os.path
import pathlib
import warnings
from collections import deque
from collections.abc import Iterator
from functools import partial
from typing import Self, Literal, overload, Any, TypedDict, Iterable

import numpy as np
from algo_engine.base import MarketData, TickData, TradeData, TransactionData, BarData

from .volume_profile import ProfileType as VolumeProfileType, VolumeProfile
from .. import LOGGER
from ..base import GlobalStatics

__all__ = ['SamplerMode', 'SamplerData', 'MarketDataSampler', 'FixedIntervalSampler', 'FixedVolumeIntervalSampler', 'AdaptiveVolumeIntervalSampler', 'VolumeProfileSampler', 'VolumeProfileType']


class SamplerMode(enum.StrEnum):
    """
    Enum for different modes of aggregation.
    """
    update = enum.auto()
    first = enum.auto()
    accumulate = enum.auto()
    min = enum.auto()
    max = enum.auto()
    median = enum.auto()
    mean = enum.auto()
    variance = enum.auto()
    store = enum.auto()  # store the raw data, without any calculation


class SamplerData(dict):
    """
    A class for handling time-series or sampled data, supporting various aggregation modes.

    This class stores observations for multiple tickers, with the option to compute statistics
    (like mean, variance, etc.) based on the aggregation mode. Data is stored in two forms:
    'history' (past observations) and 'active' (current observations).

    Attributes:
        topic (str): The topic of the sampler.
        max_size (int): The maximum number of observations to retain in history for each ticker.
        mode (str): The mode of aggregation for the active observations. Possible values are
            from the `SamplerMode` enum (e.g., 'update', 'mean', 'median').
        auto_init (bool): If True, automatically initializes tickers when observations are added.

    Properties:
        history (dict[str, deque[float]]): A dictionary mapping ticker to deque storing historical data.
        active (dict[str, list[float]]): A dictionary mapping tickers to lists of active (current) observations.
        index (dict[str, int]): A dictionary tracking the index (position) of the latest observation for each ticker.
    """

    def __init__(
            self,
            topic: str,
            max_size: int,
            mode: SamplerMode | str = SamplerMode.update,
            auto_init: bool = True,
            history: dict[str, deque[Any]] = None,
            active: dict[str, list[Any]] = None,
            index: dict[str, int] = None,
            **kwargs
    ):
        """
        Initializes a new `SamplerData` object.

        Args:
            topic (str): The topic of the sampler.
            max_size (int): The maximum size of the history deque for each ticker.
            mode (SamplerMode | str, optional): The aggregation mode (e.g., 'update', 'mean'). Defaults to 'update'.
            auto_init (bool, optional): If True, automatically initializes tickers when observations are added. Defaults to True.
            history (dict, optional): Pre-initialized history for tickers. Defaults to None.
            active (dict, optional): Pre-initialized active observations for tickers. Defaults to None.
            index (dict, optional): Pre-initialized index for tickers. Defaults to None.
            **kwargs: Additional keyword arguments for the dictionary base class.

        Raises:
            ValueError: If `max_size` is less than or equal to 0.
            NotImplementedError: If the `mode` is not a valid `SamplerMode`.
        """
        if max_size <= 0:
            raise ValueError('max_size must be greater than 0.')

        if mode not in SamplerMode:
            raise NotImplementedError(f'Invalid mode {mode}, expect {[str(_) for _ in SamplerMode]}.')

        self.topic = topic
        self.max_size = max_size
        self.mode = str(mode)
        self.auto_init = auto_init

        super().__init__(
            history={} if history is None else history,
            active={} if active is None else active,
            index={} if index is None else index,
            **kwargs
        )

    def __len__(self):
        """
        Returns the number of tickers with history.

        Returns:
            int: The number of tickers.
        """
        return len(self.history)

    def __repr__(self):
        """
        Returns a string representation of the `SamplerData` object.

        Returns:
            str: String representation of the object.
        """
        return f'<{self.__class__.__name__}: {self.topic}>(len={self.__len__()})'

    def __contains__(self, ticker: str):
        return self.history.__contains__(ticker)

    def __iter__(self) -> Iterator[str]:
        return self.history.__iter__()

    def init(self, ticker: str, **kwargs):
        """
        Initializes the data structures for a given ticker, clearing any existing data.

        Args:
            ticker (str): The ticker to initialize.

        Returns:
            tuple[deque, list]: A tuple containing the initialized history deque and active list for the ticker.

        Raises:
            Warning: Logs a warning if the ticker already exists and will be reinitialized.
        """
        if ticker in self:
            LOGGER.warning(f'Reinitializing {ticker} in {self}! All history will be overridden!')

        history = self['history'][ticker] = deque(maxlen=self.max_size)
        active = self['active'][ticker] = []

        for key, value in kwargs.items():
            self[key][ticker] = value

        if 'index' not in kwargs:
            self['index'][ticker] = 0

        return history, active

    def clear(self):
        """
        Clears all historical, active data, and index for tickers.
        """

        # self['history'].clear()
        # self['active'].clear()
        # self['index'].clear()

        for key in self.keys():
            self[key].clear()

    def active_observation(self, ticker: str, default: Any = np.nan) -> Any:
        """
        Retrieves the current active observation for a ticker based on the selected aggregation mode.

        Args:
            ticker (str): The ticker to retrieve the active observation for.
            default (Any, optional): The default value to return if no observation is available. Defaults to `np.nan`.

        Returns:
            Any | None: The latest active observation or the computed value
                based on the aggregation mode.

        Raises:
            KeyError: If the ticker is not present in the data.
        """
        if ticker not in self:
            raise KeyError(f'{ticker} not in {self}')

        active = self['active'][ticker]

        if not active:
            return default

        match self.mode:
            case SamplerMode.update:
                latest_obs = active[-1]
            case SamplerMode.first:
                latest_obs = active[0]
            case SamplerMode.accumulate:
                latest_obs = np.sum(active)
            case SamplerMode.max:
                latest_obs = np.max(active)
            case SamplerMode.min:
                latest_obs = np.min(active)
            case SamplerMode.median:
                latest_obs = np.median(active)
            case SamplerMode.mean:
                latest_obs = np.mean(active)
            case SamplerMode.variance:
                latest_obs = np.var(active)
            case SamplerMode.store:
                latest_obs = list(active)
            case _:
                raise NotImplementedError(f'Invalid mode {self.mode}!.')

        return latest_obs

    def latest_observation(self, ticker: str, default: Any = np.nan) -> Any | None:
        """
        Retrieves the latest observation from either the active list or history.

        Args:
            ticker (str): The ticker to retrieve the latest observation for.
            default (Any, optional): The default value to return if no observation is available. Defaults to `np.nan`.

        Returns:
            Any | None: The latest observation or the default value.

        Raises:
            KeyError: If the ticker is not present in the data.
        """
        if ticker not in self:
            raise KeyError(f'{ticker} not in {self}')
        elif active := self['active'][ticker]:
            return self.active_observation(ticker=ticker, default=default)
        elif history := self.history[ticker]:
            return history[-1]
        else:
            return default

    def enroll(self, ticker: str = None, **kwargs):
        """
        Moves the latest active observation to history and clears the active list for a ticker.

        Args:
            ticker (str, optional): The ticker to enroll. If None, all tickers will be enrolled. Defaults to None.
            **kwargs: Additional arguments, such as an index to update.

        Raises:
            KeyError: If the ticker is not present in the data and auto-initialization is disabled.
        """
        if ticker is None:
            for _ticker in self['history']:
                self.enroll(ticker=_ticker)
            return

        match self.mode:
            case SamplerMode.update | SamplerMode.first | SamplerMode.max | SamplerMode.min | SamplerMode.median | SamplerMode.mean:
                latest_obs = self.latest_observation(ticker=ticker, default=np.nan)
            case SamplerMode.accumulate:
                latest_obs = self.active_observation(ticker=ticker, default=0)
            case SamplerMode.variance:
                latest_obs = self.active_observation(ticker=ticker, default=np.nan)
            case SamplerMode.store:
                latest_obs = self.active_observation(ticker=ticker, default=[])
            case _:
                raise NotImplementedError(f'Invalid mode {self.mode}!.')

        self['history'][ticker].append(latest_obs)
        self['active'][ticker].clear()
        self['index'][ticker] = kwargs['index'] if 'index' in kwargs else self['index'][ticker] + 1

    def observe(self, ticker: str, obs: float, **kwargs):
        """
        Adds a new observation to the active list for a ticker.

        Args:
            ticker (str): The ticker to observe.
            obs (float): The new observation to add.
            kwargs: Additional arguments, such as an index to update.

        Raises:
            KeyError: If the ticker is not present and auto-initialization is disabled.
        """
        if ticker not in self:
            if self.auto_init:
                self.init(ticker=ticker)
            else:
                raise KeyError(f'{ticker} not in {self}!')

        self['active'][ticker].append(obs)

        for key, value in kwargs.items():
            self[key][ticker] = value

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        """
        Converts the `SamplerData` object to a JSON string or dictionary.

        Args:
            fmt (str, optional): Format of the output ('str' or 'dict'). Defaults to 'str'.
            **kwargs: Additional arguments for `json.dumps()`.

        Returns:
            str | dict: The JSON representation of the object.

        Raises:
            ValueError: If an invalid format is provided.
        """
        data_dict = dict(self)
        data_dict.update(
            topic=self.topic,
            max_size=self.max_size,
            mode=self.mode,
            history={ticker: list(history) for ticker, history in self['history'].items()},
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}! Expected "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> Self:
        """
        Creates a `SamplerData` object from a JSON string or dictionary.

        Args:
            json_message (str | bytes | bytearray | dict): The JSON message to convert.

        Returns:
            SamplerData: The created `SamplerData` object.
        """
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        json_dict['history'] = {ticker: deque(history, maxlen=json_dict['max_size']) for ticker, history in json_dict['history'].items()}

        self = cls(**json_dict)

        return self

    @property
    def history(self) -> dict[str, deque[Any]]:
        """
        Returns the history of observations for all tickers.

        Returns:
            dict[str, deque[Any]]: A dictionary mapping tickers to historical observations.
        """
        return self['history']

    @property
    def active(self) -> dict[str, Any]:
        """
        Returns the current active observations for all tickers.

        Returns:
            dict[str, Any]: A dictionary mapping tickers to the current active observations.
        """
        active = {}

        for ticker in self:
            observation = self.active_observation(ticker=ticker, default=None)

            if observation is None:
                continue

            active[ticker] = observation

        return active

    @property
    def latest(self) -> dict[str, Any]:
        """
        Returns the latest active observations for all tickers.

        Returns:
            dict[str, Any]: A dictionary mapping tickers to the latest observations.
        """
        latest = {}

        for ticker in self:
            observation = self.latest_observation(ticker=ticker, default=None)

            if observation is None:
                continue

            latest[ticker] = observation

        return latest

    @property
    def index(self) -> dict[str, int]:
        """
        Returns the current index for all tickers.

        Returns:
            dict[str, int]: A dictionary mapping tickers to their current index positions.
        """
        return self['index']


class MarketDataSampler(object, metaclass=abc.ABCMeta):
    """
    Abstract base class for handling sampled market data using multiple `SamplerData` instances.

    This class provides methods for managing and retrieving time-series data for various topics (such as stock tickers).
    It supports different aggregation modes (like `mean`, `max`, `min`, etc.) through `SamplerData`.

    Attributes:
        sampler_data (dict[str, SamplerData]): A dictionary mapping topics (as strings) to their associated `SamplerData` instances.

    Methods:
        register_sampler: Registers a new sampler for a given topic with specified parameters.
        get_history: Retrieves the history of observations for a given topic, optionally filtering by ticker.
        get_active: Retrieves the active observation for a given topic, optionally filtering by ticker.
        get_latest: Retrieves the latest observation for a given topic, optionally filtering by ticker.
        get_sampler: Deprecated. Use `get_history()` or `get_active()` instead.
        log_obs: Abstract method that logs observations for a ticker.
        on_triggered: Callback function for sampler triggered, to be overridden.
        to_json: Serializes the sampler data into JSON format.
        from_json: Deserializes JSON data into an instance of this class.
        clear: Clears all registered samplers and their data.
    """

    def __init__(self):
        self.sampler_data: dict[str, SamplerData] = getattr(self, 'sampler_data', {})  # to avoid de-reference the dict during nested inheritance initialization.
        self.contexts: dict = getattr(self, 'contexts', {})

    def register_sampler(self, topic: str, max_size: int, mode: SamplerMode | Literal['update', 'accumulate', 'min', 'max', 'median'], **kwargs) -> SamplerData:
        """
        Registers a new sampler with the specified topic and parameters.

        Args:
            topic (str): The topic for the sampler.
            max_size (int): The maximum size of the history deque.
            mode (SamplerMode | Literal['update', 'accumulate', 'min', 'max', 'median']): The aggregation mode for the sampler.
            **kwargs: Additional keyword arguments for `SamplerData`.

        Returns:
            SamplerData: The registered `SamplerData` instance.

        Raises:
            DeprecationWarning: If mode is not explicitly set.
        """
        if mode is None:
            mode = SamplerMode.update
            warnings.warn(f'Mode should be set explicitly. Default value is {SamplerMode.update}!', DeprecationWarning, stacklevel=2)

        if topic in self.sampler_data:
            sampler_data = self.sampler_data[topic]
            LOGGER.warning(f'Sampler {sampler_data} already registered in {self.__class__.__name__}!')
            return sampler_data

        sampler_data = self.sampler_data[topic] = SamplerData(
            topic=topic,
            max_size=max_size,
            mode=mode,
            auto_init=kwargs.get('auto_init', True),
            history=kwargs.get('history'),
            active=kwargs.get('active'),
            index=kwargs.get('index'),
            **kwargs
        )

        return sampler_data

    @overload
    def get_history(self, topic: str) -> dict[str, deque[Any]]:
        ...

    @overload
    def get_history(self, topic: str, ticker: str) -> deque[Any]:
        ...

    def get_history(self, topic: str, ticker: str = None):
        """
        Retrieves historical data for a specific topic or ticker.

        Args:
            topic (str): The topic to retrieve data for.
            ticker (str, optional): The specific ticker to retrieve data for. Defaults to None.

        Returns:
            dict[str, deque[Any]] | deque[Any]:
            A dictionary of historical data for all tickers if ticker is None, or a deque of historical data for a specific ticker.

        Raises:
            KeyError: If the topic or ticker is not found in the data.
        """
        if topic not in self.sampler_data:
            raise KeyError(f'Topic {topic} not found in {self.__class__.__name__}!')

        sampler_data = self.sampler_data[topic]

        if ticker is None:
            return sampler_data.history
        elif ticker not in sampler_data:
            raise KeyError(f'ticker {ticker} not found in {self.__class__.__name__}.{sampler_data}!')
        else:
            return sampler_data.history[ticker]

    @overload
    def get_active(self, topic: str) -> dict[str, Any]:
        ...

    @overload
    def get_active(self, topic: str, ticker: str) -> Any:
        ...

    def get_active(self, topic: str, ticker: str = None):
        """
        Retrieves the current active observations for a specific topic or ticker.

        Args:
            topic (str): The topic to retrieve data for.
            ticker (str, optional): The specific ticker to retrieve data for. Defaults to None.

        Returns:
            dict[str, Any] | Any:
            A dictionary of active observations for all tickers if ticker is None, or the active observation for a specific ticker.

        Raises:
            KeyError: If the topic or ticker is not found in the data.
        """
        if topic not in self.sampler_data:
            raise KeyError(f'Topic {topic} not found in {self.__class__.__name__}!')

        sampler_data = self.sampler_data[topic]

        if ticker is None:
            return sampler_data.active
        elif ticker not in sampler_data:
            raise KeyError(f'ticker {ticker} not found in {self.__class__.__name__}.{sampler_data}!')
        else:
            return sampler_data.active_observation(ticker=ticker)

    @overload
    def get_latest(self, topic: str) -> dict[str, Any]:
        ...

    @overload
    def get_latest(self, topic: str, ticker: str) -> Any:
        ...

    def get_latest(self, topic: str, ticker: str = None):
        """
        Retrieves the latest observation for a specific topic or ticker. If there is an active observation, use the active one, or fallback to the last historical one.

        Args:
            topic (str): The topic to retrieve data for.
            ticker (str, optional): The specific ticker to retrieve data for. Defaults to None.

        Returns:
            dict[str, Any] | Any:
            A dictionary of latest observations for all tickers if ticker is None, or the latest observation for a specific ticker.

        Raises:
            KeyError: If the topic or ticker is not found in the data.
        """
        if topic not in self.sampler_data:
            raise KeyError(f'Topic {topic} not found in {self.__class__.__name__}!')

        sampler_data = self.sampler_data[topic]

        if ticker is None:
            return sampler_data.latest
        elif ticker not in sampler_data:
            raise KeyError(f'ticker {ticker} not found in {self.__class__.__name__}.{sampler_data}!')
        else:
            return sampler_data.latest_observation(ticker=ticker)

    def get_sampler(self, topic: str) -> dict[str, list[Any]]:
        """
        Deprecated method that retrieves both historical data and the latest active observation for each ticker.

        Args:
            topic (str): The topic to retrieve the sampler data for.

        Returns:
            dict[str, list[Any]]: A dictionary of tickers with lists of historical data and the latest observation.

        Raises:
            DeprecationWarning: This method is deprecated.
        """
        warnings.warn('get_sampler() is deprecated, use get_history() or get_active() instead.', DeprecationWarning, stacklevel=2)

        sampler_data = self.sampler_data[topic]
        sampler = dict()

        for ticker in sampler_data:
            obs = list(sampler_data.history[ticker])
            obs.append(sampler_data.active_observation(ticker=ticker))
            sampler[ticker] = obs

        return sampler

    @abc.abstractmethod
    def log_obs(self, ticker: str, timestamp: float, observation: dict[str, ...] = None, **kwargs):
        """
        Logs observations. This method must be implemented by subclasses.

        Args:
            ticker (str): The ticker of the observation.
            timestamp (float): The timestamp of the observation.
            observation (dict[str, ...], optional): Additional observation data.
            **kwargs: Additional arguments for logging.
        """
        ...

    def on_triggered(self, ticker: str, topic: str, sampler: SamplerData, **kwargs):
        """
        Placeholder method that can be overridden by subclasses to handle specific events.

        Args:
            ticker (str): The ticker related to the event.
            topic (str): The topic related to the event.
            sampler (SamplerData): The SamplerData instance related to the event.
            **kwargs: Additional arguments for handling the event.
        """
        pass

    def to_json(self, fmt: str = 'str', **kwargs) -> str | dict:
        """
        Converts the MarketDataSampler instance to JSON format.

        Args:
            fmt (str, optional): The format of the output ('str' or 'dict'). Defaults to 'str'.
            **kwargs: Additional arguments for `json.dumps()`.

        Returns:
            str | dict: The JSON representation of the instance.

        Raises:
            ValueError: If an invalid format is provided.
        """
        data_dict = dict(
            sampler_data={topic: sampler_data.to_json(fmt='dict') for topic, sampler_data in self.sampler_data.items()},
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    @abc.abstractmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> Self:
        """
        Creates an instance of the class from a JSON representation.

        Args:
            json_message (str | bytes | bytearray | dict): The JSON representation of the instance.

        Returns:
            Self: An instance of the class.

        Raises:
            ValueError: If the JSON message is invalid.
        """
        ...

    def clear(self):
        """
        Clears all stored data.

        This method will require the samplers to be registered again if they need to be used after calling this method.
        """
        for topic, sampler_data in self.sampler_data.items():
            for ticker, dq in sampler_data['history'].items():
                dq.clear()

            self.sampler_data[topic]['index'].clear()

        # using this code will require the sampler to be registered again.
        self.sampler_data.clear()


class FixedIntervalSampler(MarketDataSampler, metaclass=abc.ABCMeta):
    """
    Abstract base class for a fixed interval sampler designed for temporal data sampling.

    Attributes:
        sampling_interval (float): Time interval between consecutive samples in seconds.
        sample_size (int): Number of samples to be stored in the sampler.

    Notes:
        `sampling_interval` must be a positive value; otherwise, a warning is issued.
        `sample_size` must be greater than 2 according to Shannon's Theorem.
        The sampler is NOT guaranteed to be triggered in a fixed interval, if corresponding market data is missing (e.g. in a melt down event or hit the upper bound limit.)
        To partially address this issue, subscribe TickData (market snapshot) from exchange and use it to update the sampler.
    """

    def __init__(self, sampling_interval: float = 1., sample_size: int = 60):
        """
        Initializes a FixedIntervalSampler instance.

        Args:
            sampling_interval (float): Time interval between consecutive samples, in seconds. Default is 1 second.
            sample_size (int): Number of samples to store in the sampler. Default is 60 samples.

        Raises:
            Warning: If `sampling_interval` is not a positive value, a warning is issued.
            Warning: If `sample_size` is less than or equal to 2, a warning is issued.
        """
        self.sampling_interval = sampling_interval
        self.sample_size = sample_size

        # Warning for sampling_interval
        if sampling_interval <= 0:
            LOGGER.warning(f'{self.__class__.__name__} should have a positive sampling_interval')

        # Warning for sample_interval by Shannon's Theorem
        if sample_size <= 2:
            LOGGER.warning(f"{self.__class__.__name__} should have a larger sample_size, by Shannon's Theorem, sample_size should be greater than 2")

        super().__init__()
        self.contexts['idx']: dict[str, int] = {}

    def register_sampler(self, topic: str, mode: SamplerMode | Literal['update', 'accumulate', 'min', 'max', 'median'], **kwargs) -> dict:
        """
        Registers a sampler for a specific topic with a specified mode and additional options.

        Args:
            topic (str): The topic to register the sampler under.
            mode (SamplerMode | Literal['update', 'accumulate', 'min', 'max', 'median']): The mode in which the sampler operates.
            **kwargs: Additional options for configuring the sampler.

        Returns:
            dict: A dictionary representing the registered sampler data.
        """
        sampler_data = super().register_sampler(
            topic=topic,
            max_size=self.sample_size,
            mode=mode,
            **kwargs
        )

        return sampler_data

    def log_obs(self, ticker: str, timestamp: float, observation: dict[str, ...] = None, **kwargs):
        """
        Logs an observation for a given ticker at the specified timestamp.

        Send all registered observations at the same time to the sampler is advised.

        Args:
            ticker (str): The ticker for which the observation is being logged.
            timestamp (float): The timestamp at which the observation occurred.
            observation (dict[str, ...], optional): A dictionary containing observation data.
            **kwargs: Additional observation data that can be included.

        Raises:
            ValueError: If the topic for the observation is not found in the sampler data.
        """
        observation_copy = {}

        if observation is not None:
            observation_copy.update(observation)

        observation_copy.update(kwargs)

        last_idx = self.contexts['idx'].get(ticker, 0)
        idx = self.contexts['idx'][ticker] = timestamp // self.sampling_interval

        for topic, obs_value in observation_copy.items():
            if topic not in self.sampler_data:
                raise ValueError(f'Invalid sampler topic: {topic}! Expected topic {list(self.sampler_data)}.')

            sampler_data = self.sampler_data[topic]

            if ticker not in sampler_data:
                sampler_data.init(ticker=ticker)

            if last_idx == 0:
                sampler_data.observe(ticker=ticker, obs=obs_value, index=idx)
            if idx > last_idx:
                sampler_data.enroll(ticker=ticker, index=idx)
                sampler_data.observe(ticker=ticker, obs=obs_value, index=idx)
                self.on_triggered(ticker=ticker, topic=sampler_data.topic, sampler=sampler_data)
            else:
                sampler_data.observe(ticker=ticker, obs=obs_value, index=idx)

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = super().to_json(fmt='dict', **kwargs)
        data_dict.update(
            sampling_interval=self.sampling_interval,
            sample_size=self.sample_size
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            sampling_interval=json_dict['sampling_interval'],
            sample_size=json_dict['sample_size']
        )

        for topic, sampler_json in json_dict['sampler_data'].items():
            sampler_data = SamplerData.from_json(sampler_json)
            self.sampler_data[topic] = sampler_data

        return self


class FixedVolumeIntervalSampler(FixedIntervalSampler, metaclass=abc.ABCMeta):
    """
    Concrete implementation of FixedIntervalSampler with fixed volume interval sampling.

    Args:
        sampling_interval (float): Volume interval between consecutive samples. Default is 100.
        sample_size (int): Number of samples to be stored. Default is 20.

    Attributes:
        sampling_interval (float): Volume interval between consecutive samples.
        sample_size (int): Number of samples to be stored.

    Methods:
        accumulate_volume(ticker: str = None, volume: float = 0., market_data: MarketData = None, use_notional: bool = False)
            Accumulates volume based on market data or explicit ticker and volume.

        log_obs(ticker: str, value: float, storage: Dict[str, Dict[float, float]], volume_accumulated: float = None)
            Logs an observation for the given ticker at the specified volume-accumulated timestamp.

        clear()
            Clears all stored data, including accumulated volumes.

    Notes:
        This class extends FixedIntervalSampler and provides additional functionality for volume accumulation.
        The sampling_interval is in shares, representing the fixed volume interval for sampling.

    """

    def __init__(self, sampling_interval: float = 100., sample_size: int = 20, use_notional: bool = True):
        """
        Initialize the FixedVolumeIntervalSampler.

        Parameters:
        - sampling_interval (float): Volume interval between consecutive samples. Default is 100.
        - sample_size (int): Number of samples to be stored. Default is 20.
        """
        super().__init__(sampling_interval=sampling_interval, sample_size=sample_size)

        self.contexts['vol_acc']: dict[str, float] = {}
        self.contexts['use_notional']: bool = use_notional

    @overload
    def accumulate_volume(self, ticker: str, volume: float):
        ...

    @overload
    def accumulate_volume(self, market_data: MarketData):
        ...

    def accumulate_volume(self, *, ticker: str = None, volume: float = 0., market_data: MarketData = None):
        """
        Accumulates volume based on market data or explicit ticker and volume.

        Parameters:
        - ticker (str): Ticker symbol for volume accumulation.
        - volume (float): Volume to be accumulated.
        - market_data (MarketData): Market data for dynamic volume accumulation.
        - use_notional (bool): Flag indicating whether to use notional instead of volume.

        Raises:
        - NotImplementedError: If market data type is not supported.

        """
        if market_data is not None and (not GlobalStatics.PROFILE.is_market_session(market_data.timestamp)):
            return

        use_notional = self.contexts['use_notional']
        if market_data is not None and isinstance(market_data, (TradeData, TransactionData)):
            ticker = market_data.ticker
            volume = market_data.notional if use_notional else market_data.volume

            self.contexts['vol_acc'][ticker] = self.contexts['vol_acc'].get(ticker, 0.) + volume
        elif isinstance(market_data, TickData):
            ticker = market_data.ticker
            vol_acc = market_data.total_traded_notional if use_notional else market_data.total_traded_volume

            if vol_acc is not None and np.isfinite(vol_acc) and vol_acc:
                self.contexts['vol_acc'][ticker] = vol_acc
        elif isinstance(market_data, BarData):
            ticker = market_data.ticker
            volume = market_data.notional if use_notional else market_data.volume

            self.contexts['vol_acc'][ticker] = self.contexts['vol_acc'].get(ticker, 0.) + volume
        elif market_data is not None:
            raise NotImplementedError(f'Can not handle market data type {market_data.__class__}, expect TickData, BarData, TradeData and TransactionData.')
        else:
            if ticker is not None:
                self.contexts['vol_acc'][ticker] = self.contexts['vol_acc'].get(ticker, 0.) + volume
            else:
                raise ValueError('Must assign market_data, or ticker and volume')

    def log_obs(self, ticker: str, timestamp: float, observation: dict[str, ...] = None, **kwargs):
        """
        Logs an observation for the given ticker at the specified volume-accumulated timestamp.

        Args:
            ticker (str): The ticker for which the observation is being logged.
            timestamp (float): The accumulated trading volume or notional.
            observation (dict[str, ...], optional): A dictionary containing observation data.
            **kwargs: Additional observation data that can be included.

        """
        super().log_obs(
            ticker=ticker,
            timestamp=self.contexts['vol_acc'].get(ticker, 0.),
            observation=observation,
            **kwargs
        )

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = super().to_json(fmt='dict')

        data_dict.update(
            vol_acc=self.contexts['vol_acc'],
            use_notional=self.contexts['use_notional']
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            sampling_interval=json_dict['sampling_interval'],
            sample_size=json_dict['sample_size']
        )

        for sampler_json in json_dict['sampler_data']:
            sampler_data = SamplerData.from_json(sampler_json)
            self.sampler_data[sampler_data.topic] = sampler_data

        self.contexts['vol_acc'].update(json_dict['accumulated_volume'])

        return self

    def clear(self):
        """
        Clears all stored data, including accumulated volumes.
        """
        super().clear()

        self.contexts['vol_acc'].clear()


class AdaptiveVolumeIntervalSampler(FixedVolumeIntervalSampler, metaclass=abc.ABCMeta):
    """
    Update the FixedVolumeIntervalSampler with adaptive volume interval.
    The volume interval is estimated using the trading volume in opening of the market.

    Args:
        sampling_interval (float): Temporal interval between consecutive samples for generating the baseline. Default is 60.
        sample_size (int): Number of samples to be stored. Default is 20.
        baseline_window (int): Number of observations used for baseline calculation. Default is 100.

    Attributes:
        sampling_interval (float): Temporal interval between consecutive samples for generating the baseline.
        sample_size (int): Number of samples to be stored.
        baseline_window (int): Number of observations used for baseline calculation.

    Methods:
        _update_volume_baseline(ticker: str, timestamp: float, volume_accumulated: float = None) -> float | None
            Updates and calculates the baseline volume for a given ticker.

        log_obs(ticker: str, value: float, storage: Dict[str, Dict[Tuple[float, float], float]], volume_accumulated: float = None, timestamp: float = None, allow_oversampling: bool = False)
            Logs an observation for the given ticker at the specified volume-accumulated timestamp.

        clear()
            Clears all stored data, including accumulated volumes and baseline information.

    Notes:
        The class designed to work with multi-inheritance.
        The sampling_interval is a temporal interval in seconds for generating the baseline.
        The baseline is calculated adaptively based on the provided baseline_window.

    """

    class VolumeBaseline(TypedDict):
        baseline: dict[str, float]
        sampling_interval: dict[str, float]
        obs_vol_acc_start: dict[str, float]
        obs_index: dict[float, float]
        obs_vol_acc: dict[str, deque]

    def __init__(self, sampling_interval: float = 60., sample_size: int = 20, baseline_window: int = 100, aligned_interval: bool = True, use_notional: bool = True):
        """
        Initialize the AdaptiveVolumeIntervalSampler.

        Parameters:
        - sampling_interval (float): Temporal interval between consecutive samples for generating the baseline. Default is 60.
        - sample_size (int): Number of samples to be stored. Default is 20.
        - baseline_window (int): Number of observations used for baseline calculation. Default is 100.
        - aligned (bool): Whether the sampling of each ticker is aligned (same temporal interval)
        """
        super().__init__(sampling_interval=sampling_interval, sample_size=sample_size, use_notional=use_notional)
        self.baseline_window = baseline_window
        self.aligned_interval = aligned_interval

        self.contexts['vol_baseline'] = self.VolumeBaseline(
            baseline={},
            sampling_interval={},
            obs_vol_acc_start={},
            obs_index={},
            obs_vol_acc={}
        )

    def register_sampler(self, topic: str, mode: SamplerMode | Literal['update', 'accumulate', 'min', 'max', 'median'], **kwargs) -> dict:
        sampler_data = super().register_sampler(topic=topic, mode=mode, index_vol={}, **kwargs)
        return sampler_data

    def _update_volume_baseline(self, ticker: str, timestamp: float, volume_accumulated: float = None, min_obs: int = None, auto_register: bool = True) -> float | None:
        """
        Updates and calculates the baseline volume for a given ticker.

        Parameters:
        - ticker (str): Ticker symbol for volume baseline calculation.
        - timestamp (float): Timestamp of the observation.
        - volume_accumulated (float): Accumulated volume at the provided timestamp.

        Returns:
        - float | None: Calculated baseline volume or None if the baseline is not ready.

        """
        min_obs = min(self.sample_size, int(self.baseline_window // 2)) if min_obs is None else min_obs

        volume_baseline_dict: dict[str, float] = self.contexts['vol_baseline']['baseline']
        obs_vol_acc_start_dict: dict[str, float] = self.contexts['vol_baseline']['obs_vol_acc_start']
        obs_index_dict: dict[str, float] = self.contexts['vol_baseline']['obs_index']

        volume_baseline = volume_baseline_dict.get(ticker, np.nan)
        volume_accumulated = self.contexts['vol_acc'].get(ticker, 0.) if volume_accumulated is None else volume_accumulated

        if np.isfinite(volume_baseline):
            return volume_baseline

        if ticker in (_ := self.contexts['vol_baseline']['obs_vol_acc']):
            obs_vol_acc: deque = _[ticker]
        elif auto_register:
            obs_vol_acc = _[ticker] = deque(maxlen=self.baseline_window)
        else:
            LOGGER.warning(f'Ticker {ticker} not registered in {self.__class__.__name__}, perhaps the subscription has changed?')
            return None

        if not obs_vol_acc:
            # in this case, one obs of the trade data will be missed
            obs_start_acc_vol = obs_vol_acc_start_dict[ticker] = volume_accumulated
        else:
            obs_start_acc_vol = obs_vol_acc_start_dict[ticker]

        last_idx = obs_index_dict.get(ticker, 0)
        obs_idx = obs_index_dict[ticker] = timestamp // self.sampling_interval
        obs_ts = obs_idx * self.sampling_interval
        volume_acc = volume_accumulated - obs_start_acc_vol
        baseline_ready = False

        if not obs_vol_acc:
            obs_vol_acc.append(volume_acc)
        elif obs_idx == last_idx:
            obs_vol_acc[-1] = volume_acc
        # in this case, the obs_vol_acc is full, and a new index received, baseline is ready
        elif len(obs_vol_acc) == self.baseline_window:
            baseline_ready = True
        else:
            obs_vol_acc.append(volume_acc)

        # convert vol_acc to vol
        obs_vol = []
        vol_acc_last = 0.
        for vol_acc in obs_vol_acc:
            obs_vol.append(vol_acc - vol_acc_last)
            vol_acc_last = vol_acc

        if len(obs_vol) < min_obs:
            baseline_est = None
        else:
            if baseline_ready:
                baseline_est = np.mean(obs_vol)
            else:
                obs_vol_history = obs_vol[:-1]
                obs_vol_current_adjusted = obs_vol[-1] / max(1., timestamp - obs_ts) * self.sampling_interval

                if timestamp - obs_ts > self.sampling_interval * 0.5 and obs_vol_current_adjusted is not None:
                    obs_vol_history.append(obs_vol_current_adjusted)

                baseline_est = np.mean(obs_vol_history) if obs_vol_history else None

        if baseline_ready:
            if np.isfinite(baseline_est) and baseline_est > 0:
                self.contexts['vol_baseline']['baseline'][ticker] = baseline_est
            else:
                LOGGER.error(f'{ticker} Invalid estimated baseline {baseline_est}, observation window extended.')
                obs_vol_acc.clear()
                self.contexts['vol_baseline']['obs_vol_acc_start'].pop(ticker)
                self.contexts['vol_baseline']['sampling_interval'].pop(ticker)

        if baseline_est is not None:
            self.contexts['vol_baseline']['sampling_interval'][ticker] = baseline_est

        return baseline_est

    def log_obs(self, ticker: str, timestamp: float, volume_accumulated: float = None, observation: dict[str, ...] = None, auto_register: bool = True, **kwargs):
        """
        Logs an observation for the given ticker at the specified volume-accumulated timestamp.

        Parameters:
        - ticker (str): Ticker symbol for the observation.
        - value (float): Value of the observation.
        - storage (Dict[str, Dict[Tuple[float, float], float]]): Storage dictionary for sampled observations.
        - volume_accumulated (float): Accumulated volume for the observation timestamp.
        - timestamp (float): Timestamp of the observation.
        - allow_oversampling (bool): Flag indicating whether oversampling is allowed.

        Raises:
        - ValueError: If timestamp is not provided.

        """
        # step 0: copy the observation
        observation_copy = {}

        if observation is not None:
            observation_copy.update(observation)

        observation_copy.update(kwargs)

        if volume_accumulated is None:
            volume_accumulated = self.contexts['vol_acc'].get(ticker, 0.)

        # step 1: calculate sampling interval
        volume_sampling_interval = self._update_volume_baseline(ticker=ticker, timestamp=timestamp, volume_accumulated=volume_accumulated, auto_register=auto_register)

        # step 2: calculate index
        if volume_sampling_interval is None:
            # baseline still in generating process, fallback to fixed temporal interval mode
            idx_ts = timestamp // self.sampling_interval
            idx_vol = 0
        elif volume_sampling_interval <= 0 or not np.isfinite(volume_sampling_interval):
            LOGGER.error(f'Invalid volume update interval for {ticker}, expect positive float, got {volume_sampling_interval}')
            return
        elif self.aligned_interval:
            volume_sampling_interval = 0.
            volume_accumulated = 0.

            volume_baseline = self.contexts['vol_baseline']['baseline']
            sampling_interval = self.contexts['vol_baseline']['sampling_interval']
            weights = getattr(self, 'weights', {})

            for component, vol_acc in self.contexts['vol_acc'].items():
                weight = weights.get(component, 0.)
                vol_sampling_interval = volume_baseline.get(component, sampling_interval.get(component, 0.))

                if not weight:
                    continue

                if not vol_sampling_interval:
                    continue

                volume_accumulated += vol_acc * weight
                volume_sampling_interval += vol_sampling_interval * weight

            idx_ts = 0.
            idx_vol = volume_accumulated // volume_sampling_interval
        else:
            idx_ts = 0.
            idx_vol = volume_accumulated // volume_sampling_interval

        # step 3: update sampler
        for topic, obs_value in observation_copy.items():
            if topic not in self.sampler_data:
                raise ValueError(f'Invalid sampler topic: {topic}! Expected topic {list(self.sampler_data)}.')

            sampler_data = self.sampler_data[topic]

            if ticker not in sampler_data:
                sampler_data.init(ticker=ticker, index_vol=0)

            last_idx_ts = sampler_data['index'][ticker]
            last_idx_vol = sampler_data['index_vol'][ticker]

            if last_idx_ts == last_idx_vol == 0:
                sampler_data.observe(ticker=ticker, obs=obs_value, index=idx_ts, index_vol=idx_vol)
            elif idx_ts > last_idx_ts or idx_vol > last_idx_vol:
                sampler_data.enroll(ticker=ticker)
                sampler_data.observe(ticker=ticker, obs=obs_value, index=idx_ts, index_vol=idx_vol)
                self.on_triggered(ticker=ticker, topic=topic, sampler=sampler_data)
            else:
                sampler_data.observe(ticker=ticker, obs=obs_value)

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict: dict = super().to_json(fmt='dict')

        data_dict.update(
            baseline_window=self.baseline_window,
            aligned_interval=self.aligned_interval,
            volume_baseline=dict(
                baseline=self.contexts['vol_baseline']['baseline'],
                sampling_interval=self.contexts['vol_baseline']['sampling_interval'],
                obs_vol_acc_start=self.contexts['vol_baseline']['obs_vol_acc_start'],
                obs_index=self.contexts['vol_baseline']['obs_index'],
                obs_vol_acc={ticker: list(dq) for ticker, dq in self.contexts['vol_baseline']['obs_vol_acc'].items()}
            )
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            sampling_interval=json_dict['sampling_interval'],
            sample_size=json_dict['sample_size'],
            baseline_window=json_dict['baseline_window'],
            aligned_interval=json_dict['aligned_interval']
        )

        for topic, sampler_json in json_dict['sampler_data'].items():
            sampler_data = SamplerData.from_json(sampler_json)
            self.sampler_data[topic] = sampler_data

        self.contexts['vol_acc'].update(json_dict['accumulated_volume'])

        self.contexts['vol_baseline']['baseline'].update(json_dict['volume_baseline']['baseline'])
        self.contexts['vol_baseline']['sampling_interval'].update(json_dict['volume_baseline']['sampling_interval'])
        self.contexts['vol_baseline']['obs_vol_acc_start'].update(json_dict['volume_baseline']['obs_vol_acc_start'])
        self.contexts['vol_baseline']['obs_index'].update(json_dict['volume_baseline']['obs_index'])
        for ticker, data in json_dict['volume_baseline']['obs_vol_acc'].items():
            if ticker in self.contexts['vol_baseline']['obs_vol_acc']:
                self.contexts['vol_baseline']['obs_vol_acc'][ticker].extend(data)
            else:
                self.contexts['vol_baseline']['obs_vol_acc'][ticker] = deque(data, maxlen=self.baseline_window)

        return self

    def clear(self):
        """
        Clears all stored data, including accumulated volumes and baseline information.
        """
        super().clear()

        self.contexts['vol_baseline']['baseline'].clear()
        self.contexts['vol_baseline']['sampling_interval'].clear()
        self.contexts['vol_baseline']['obs_vol_acc_start'].clear()
        self.contexts['vol_baseline']['obs_index'].clear()
        self.contexts['vol_baseline']['obs_vol_acc'].clear()

    @property
    def baseline_ready(self) -> bool:
        subscription = set(self.contexts['vol_baseline']['obs_vol_acc'])

        for ticker in subscription:
            sampling_interval = self.contexts['vol_baseline']['sampling_interval'].get(ticker, np.nan)

            if not np.isfinite(sampling_interval):
                return False

        return True

    @property
    def baseline_stable(self) -> bool:
        subscription = set(self.contexts['vol_baseline']['obs_vol_acc'])

        for ticker in subscription:
            sampling_interval = self.contexts['vol_baseline']['baseline'].get(ticker, np.nan)

            if not np.isfinite(sampling_interval):
                return False

        return True


class VolumeProfileSampler(FixedVolumeIntervalSampler, metaclass=abc.ABCMeta):
    def __init__(self, sampling_interval: float = 60., sample_size: int = 20, profile_type: VolumeProfileType = 'simple_online', use_notional: bool = True, **kwargs):
        """
        Initialize the AdaptiveVolumeIntervalSampler.

        Parameters:
        - sampling_interval (float): Temporal interval between consecutive samples for generating the baseline. Default is 60.
        - sample_size (int): Number of samples to be stored. Default is 20.
        - baseline_window (int): Number of observations used for baseline calculation. Default is 100.
        - aligned (bool): Whether the sampling of each ticker is aligned (same temporal interval)
        """
        super().__init__(sampling_interval=sampling_interval, sample_size=sample_size, use_notional=use_notional)

        profile_type = VolumeProfileType(profile_type)
        profile_config = {'use_notional': use_notional}
        constructor = profile_type.get_profile()

        for name, param in inspect.signature(constructor).parameters.items():
            if name == 'ticker':
                continue

            if name in kwargs:
                profile_config[name] = kwargs[name]
            elif param.default is param.empty and name not in kwargs:
                raise ValueError(f'{self.__class__.__name__} required parameter {name} unfilled!')

        self.contexts['profile_type'] = profile_type
        self.contexts['profile_config'] = profile_config
        self.contexts['profile_constructor'] = partial(constructor, **profile_config)
        self.contexts['estimated_volume_interval']: dict[str, float] = {}

        self.volume_profile: dict[str, VolumeProfile] = {}

    def initialize_volume_profile(self, subscription: Iterable[str], data: dict[str, list[MarketData]] = None, profile_file: dict[str, str | pathlib.Path] = None, **kwargs):
        for ticker in subscription:
            # try using the profile file cache
            if profile_file and profile_file.get(ticker):
                _profile_file = profile_file[ticker]
                if os.path.isfile(_profile_file):
                    with open(_profile_file, 'r') as f:
                        self.volume_profile[ticker] = VolumeProfile.from_json(json.load(f))
                        LOGGER.info(f'Volume profile of {ticker} loaded from {_profile_file}.')
                        continue

            profile_type = self.contexts['profile_type']
            profile_config = self.contexts['profile_config']
            volume_profile = profile_type.to_profile(ticker=ticker, **profile_config)

            # try loading market_data for the profile
            if data and data.get(ticker, []):
                market_data_list = data[ticker]
                LOGGER.info(f'Using {len(market_data_list)} market data to initialize {ticker} volume profile.')
                volume_profile.fit(data=market_data_list)
            elif profile_type != VolumeProfileType.simple_online:
                LOGGER.warning(f'Volume profile of {ticker} not initialized! Use with caution!')

            self.volume_profile[ticker] = volume_profile

    def accumulate_volume(self, *, ticker: str = None, volume: float = 0., market_data: MarketData = None):
        assert market_data is not None, f"{self.__class__.__name__}.accumulate_volume requires market_data input!"

        super().accumulate_volume(market_data=market_data)

        if market_data.ticker in self.volume_profile:
            self.volume_profile[market_data.ticker].on_data(market_data=market_data)

    def log_obs(self, ticker: str, timestamp: float, volume_accumulated: float = None, observation: dict[str, ...] = None, auto_register: bool = True, **kwargs):
        # step 0: copy the observation
        observation_copy = {}

        if observation is not None:
            observation_copy.update(observation)

        observation_copy.update(kwargs)

        if volume_accumulated is None:
            volume_accumulated = self.contexts['vol_acc'].get(ticker, 0.)

        volume_profile = self.volume_profile[ticker]

        if ticker in self.contexts['estimated_volume_interval']:
            estimated_volume_interval = self.contexts['estimated_volume_interval'][ticker]
        else:
            estimated_volume_ttl = volume_profile.predict()
            estimated_n_interval = volume_profile.session_length / self.sampling_interval
            estimated_volume_interval = self.contexts['estimated_volume_interval'][ticker] = estimated_volume_ttl / estimated_n_interval

        # replace the vol_idx with ts_idx
        if estimated_volume_interval is None or np.isnan(estimated_volume_interval):
            idx = int(volume_profile.session_ts(datetime.datetime.fromtimestamp(timestamp, tz=GlobalStatics.PROFILE.time_zone).time()) // self.sampling_interval)
        else:
            idx = volume_accumulated // estimated_volume_interval

        # step 3: update sampler
        for topic, obs_value in observation_copy.items():
            if topic not in self.sampler_data:
                raise ValueError(f'Invalid sampler topic: {topic}! Expected topic {list(self.sampler_data)}.')

            sampler_data = self.sampler_data[topic]

            if ticker not in sampler_data:
                sampler_data.init(ticker=ticker, index=0)

            last_idx = sampler_data['index'][ticker]

            if last_idx >= idx:
                sampler_data.observe(ticker=ticker, obs=obs_value, index=idx)
            else:
                sampler_data.enroll(ticker=ticker)
                sampler_data.observe(ticker=ticker, obs=obs_value, index=idx)
                self.on_triggered(ticker=ticker, topic=topic, sampler=sampler_data)
                self.contexts['estimated_volume_interval'].pop(ticker, None)

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict: dict = super().to_json(fmt='dict')

        data_dict.update(
            profile_type=self.contexts['profile_type'].name,
            profile_config=self.contexts['profile_config'],
            estimated_volume_interval=self.contexts['estimated_volume_interval'],
            volume_profile={ticker: volume_profile.to_json(fmt='dict') for ticker, volume_profile in self.volume_profile.items()},
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            sampling_interval=json_dict['sampling_interval'],
            sample_size=json_dict['sample_size'],
            profile_type=json_dict['profile_type'],
            # use_notional should be overloaded by profile_config
            # use_notional=json_dict['use_notional'],
            **json_dict['profile_config']
        )

        for topic, sampler_json in json_dict['sampler_data'].items():
            sampler_data = SamplerData.from_json(sampler_json)
            self.sampler_data[topic] = sampler_data

        self.contexts['estimated_volume_interval'].update(json_dict['estimated_volume_interval'])
        self.volume_profile.update({ticker: VolumeProfile.from_json(profile_dict) for ticker, profile_dict in json_dict['volume_profile'].items()})
        return self

    def clear(self):
        """
        Clears all stored data, including accumulated volumes and baseline information.
        """
        super().clear()

        self.volume_profile.clear()

    @property
    def profile_ready(self) -> bool:
        for volume_profile in self.volume_profile.values():
            if not volume_profile.is_ready:
                return False

        return True
