from __future__ import annotations

import abc
import enum
import hashlib
import inspect
import json
import pathlib
import pickle
from collections import deque
from collections.abc import Iterable
from functools import cached_property
from typing import Self, Any, overload

import numpy as np
from algo_engine.base import MarketData, TickData, TradeData, TransactionData, OrderBook, OrderData, TransactionSide
from algo_engine.engine import MarketDataMonitor, MDS

from .sampler import *
from .. import LOGGER
from ..base import GlobalStatics
from ..base.memory_core import SharedMemoryCore

LOGGER = LOGGER.getChild('Utils')
ALPHA_05 = 0.9885  # alpha = 0.5 for each minute
ALPHA_02 = 0.9735  # alpha = 0.2 for each minute
ALPHA_01 = 0.9624  # alpha = 0.1 for each minute
ALPHA_001 = 0.9261  # alpha = 0.01 for each minute
ALPHA_0001 = 0.8913  # alpha = 0.001 for each minute
ALPHA_9 = 0.1000  # window = 9, roughly 1 - ALPHA_0001
ALPHA_12 = 0.0769  # window = 12, roughly 1 - ALPHA_001
ALPHA_26 = 0.0370  # window = 26, roughly 1 - ALPHA_01


class IndexWeight(dict):
    def __init__(self, index_name: str, *args, **kwargs):
        self.index_name = index_name

        super().__init__(*args, **kwargs)

    def normalize(self):
        total_weight = sum(list(self.values()))

        if not total_weight:
            return

        for _ in self:
            self[_] /= total_weight

    def composite(self, values: dict[str, float], replace_na: float = 0.):
        weighted_sum = 0.

        for ticker, weight in self.items():
            value = values.get(ticker, replace_na)

            if np.isnan(value):
                weighted_sum += replace_na * self[ticker]
            else:
                weighted_sum += value * self[ticker]

        return weighted_sum

    @property
    def components(self) -> list[str]:
        return list(self.keys())

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            index_name=self.index_name,
            weights=dict(self),

        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> IndexWeight:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            index_name=json_dict['index_name'],
            **json_dict['weights']
        )

        return self


class FilterMode(enum.Flag):
    no_cancel = enum.auto()
    no_auction = enum.auto()
    no_order = enum.auto()
    no_trade = enum.auto()
    no_tick = enum.auto()


class FactorInfo(object):
    def __init__(self, name: str, file: str | pathlib.Path):
        self.name = name
        self.file = str(file)

        self.meta: dict[str, Any] | None = None
        self.params = None
        self.constructor: type[FactorMonitor] | None = None

        self.additional_args = []
        self.weights_required = False
        self._is_ready = False

    def _read(self) -> Self:
        self.meta = self.read_meta(file_path=self.file)
        return self

    def _parse(self) -> Self:
        if self.meta is None:
            self._read()

        for _params in self.meta['params']:
            if _params['name'] == self.name:
                self.params = _params
                break
        else:
            raise ModuleNotFoundError(f'Can not find factor {self.name} in {self.file}!')

        return self

    def _import(self) -> type[FactorMonitor]:
        import importlib.util
        import sys

        if self.meta is None:
            self._read()

        file_path = pathlib.Path(self.file)
        module_name = file_path.stem  # Get the module name from the file name

        # Load the module from the file
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)

        # Add the module to sys.modules and execute it
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Retrieve the specified name from the module
        self.constructor = getattr(module, self.meta['name'], None)
        return self.constructor

    def _inspect(self) -> Self:
        if self.params is None:
            self._parse()

        if self.constructor is None:
            self._import()

        constructor_name = self.constructor.__name__
        constructor_signature = inspect.signature(self.constructor.__init__)
        for name, param in constructor_signature.parameters.items():
            match name:
                case 'self':
                    continue
                case 'weights':
                    self.weights_required = True
                    LOGGER.info(f"<{constructor_name}>({self.name}) required argument: <weights>. Which is inferred as index weights")
                case _ if name in self.params:
                    LOGGER.info(f"<{constructor_name}>({self.name}) required argument: <{name}>. Value will be provided by factor meta.")
                case _ if param.default is param.empty:
                    self.additional_args.append((constructor_name, name, param.annotation))
                    LOGGER.warning(f'<{constructor_name}>({self.name}) required argument: <{name}>. Definition not found in parameter list! Registered into arg parser!')
                case _ if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    LOGGER.warning(f'<{constructor_name}> accepting positional arguments: <{name}>. This might causing signature inspection failure! Use with caution.')
                case _ if param.kind == inspect.Parameter.VAR_KEYWORD:
                    LOGGER.warning(f'<{constructor_name}> accepting keyword arguments: <{name}>. This might causing signature inspection failure! Use with caution.')
                case _:
                    LOGGER.info(f"<{constructor_name}>({self.name}) optional argument: <{name}>. Default value will be used.")
                    # raise SyntaxError(f'<{constructor_name}> constructor argument <{name}> can not be inferred and identified!')

        self._is_ready = True
        return self

    def load(self, override: bool = False) -> Self:
        if override or self.meta is None:
            self._read()

        if override or self.params is None:
            self._parse()

        if override or self.constructor is None:
            self._import()

        if override or not self._is_ready:
            self._inspect()

        return self

    def initialize(self, **kwargs) -> FactorMonitor:
        if not self.is_ready:
            self.load()

        params = self.params.copy()
        constructor_name = self.constructor.__name__
        constructor_signature = inspect.signature(self.constructor.__init__)

        for name, param in constructor_signature.parameters.items():
            if name == 'self':
                continue

            if name in params:
                LOGGER.info(f"{constructor_name} required argument: <{name}>. Value will be provided by factor meta.")
                continue

            if name in kwargs:
                params[name] = kwargs[name]
                LOGGER.info(f"{constructor_name} required argument: <{name}>. Provided by kwargs.")
                continue

            if param.default is param.empty:
                raise ValueError(f'{constructor_name} required argument: <{name}>. Definition not found in parameter list!')

        factor = self.constructor(**params)
        factor.contexts['meta'].update({key: value for key, value in self.meta.items() if key != 'params'})
        return factor

    @classmethod
    def read_meta(cls, file_path: str | pathlib.Path) -> dict[str, ...]:
        import ast

        with open(file_path, "r") as file:
            tree = ast.parse(file.read(), filename=file_path)

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__meta__":
                        meta_dict = eval(ast.unparse(node.value))
                        return meta_dict

        raise ImportError(f'__meta__ is not defined in {file_path}!')

    @classmethod
    def load_factor(cls, name: str, file: str | pathlib.Path) -> Self:
        self = cls(name=name, file=file)
        self.load()
        return self

    @classmethod
    def import_factor(cls, name: str, file: str | pathlib.Path) -> type[FactorMonitor]:
        self = cls(name=name, file=file)
        constructor = self._import()
        return constructor

    @classmethod
    def to_factor(cls, name: str, file: str | pathlib.Path, **kwargs) -> FactorMonitor:
        self = cls(name=name, file=file)
        factor = self.initialize(**kwargs)
        return factor

    @classmethod
    def from_file(cls, file: str | pathlib.Path) -> list[Self]:
        node_list = []
        meta_dict = cls.read_meta(file_path=file)

        for _params in meta_dict['params']:
            name = _params['name']
            node = cls(name=name, file=file)
            node.meta = meta_dict.copy()

            node.load()
            node_list.append(node)

        return node_list

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @cached_property
    def md5(self):
        with open(self.file, 'rb') as f:
            data = f.read()
            return hashlib.md5(data).hexdigest()


class FactorTree(object):
    def __init__(self, nodes: Iterable[FactorInfo] | FactorInfo = None):
        self.nodes: list[FactorInfo] = [] if nodes is None else [nodes] if isinstance(nodes, dict) else list(nodes)

    def __hash__(self) -> int:
        tree_hash = hashlib.md5()

        for node in self.nodes:
            tree_hash.update(f'{node.file}-{node.name}-{node.md5}'.encode('utf-8'))

        return int.from_bytes(tree_hash.digest())

    def digest(self) -> str:
        tree_hash = hashlib.sha256()

        for node in self.nodes:
            tree_hash.update(f'{node.file}-{node.name}-{node.md5}'.encode('utf-8'))

        return tree_hash.hexdigest()

    @overload
    def append(self, factor_info: FactorInfo) -> None:
        ...

    @overload
    def append(self, name: str, file: str | pathlib.Path) -> None:
        ...

    def append(self, *args, **kwargs) -> None:
        try:
            self._add_node(*args, **kwargs)
        except TypeError as e:
            self._append_node(*args, **kwargs)
        except Exception as e:
            raise e

    def _add_node(self, name: str, file: str | pathlib.Path) -> None:
        node = FactorInfo(name=name, file=file).load()

        if node is None:
            raise FileNotFoundError(f'Cannot find factor {name} in {file}.')

        self.nodes.append(node)

    def _append_node(self, node: FactorInfo) -> None:
        node.load()
        self.nodes.append(node)

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            dtype='FactorTree',
            nodes=[dict(name=node.name, file=node.file) for node in self.nodes],
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
            nodes=[FactorInfo(name=node_dict['name'], file=node_dict['file']).load() for node_dict in json_dict['nodes']],
        )

        return self

    def dump(self, file_path: str | pathlib.Path) -> None:
        with open(file_path, 'w') as f:
            f.write(json.dumps(self.to_json(fmt='dict'), indent=4))

    @classmethod
    def load(cls, file_path: str | pathlib.Path) -> Self:
        with open(file_path, 'r') as f:
            self = cls.from_json(json.load(f))

        return self

    def copy(self) -> Self:
        f = FactorTree([FactorInfo(name=node.name, file=node.file).load() for node in self.nodes])
        return f


class FactorMonitor(MarketDataMonitor, metaclass=abc.ABCMeta):
    def __init__(self, name: str, monitor_id: str = None, meta: dict[str, str | float | int | bool] = None, mds=None, filter_mode: FilterMode | int = FilterMode(0)):
        """
        Initializes the FactorMonitor instance.

        This method must not accept *args or **kwargs as parameters as the `from_json` function relies on the explicit signature of the constructor.

        The __init__ method of any child class must first call `FactorMonitor.__init__` to ensure that the subscription and memory core are initialized before any other class attributes.

        Additionally, child class `from_json` and `update_from_json` methods should call their respective parent implementations first.

        All parameters must be JSON serializable, as required for the `from_json` function.

        If multiprocessing is enabled, the `subscription` parameter is also required.

        Args:
            name: The name of the monitor. Must start with 'Monitor'.
            monitor_id: The ID of the monitor. If not provided, a UUID4 will be generated.
            meta: A dictionary containing metadata about the monitor. Keys must be strings, and values
                  can be of type str, float, int, or bool.
            filter_mode: A `FilterMode` flag (or integer equivalent) that specifies filtering options.
                         Defaults to `FilterMode(0)` (no filters applied).

        Notes:
            - The presence of *args or **kwargs in the constructor will raise a warning, as they are
              not allowed.
            - Parameters should be explicitly defined to maintain compatibility with the `from_json` method.
        """

        var_names = self.__class__.__init__.__code__.co_varnames

        if 'kwargs' in var_names or 'args' in var_names:
            LOGGER.warning(f'Arbitrary arguments *args and **kwargs should not be accepted by {self.__class__.__name__}.__init__ function. All parameters should be explicit.')

        assert name.startswith('Monitor')
        super().__init__(name=name, monitor_id=monitor_id)
        self.filter_mode = FilterMode(filter_mode)
        self.mds = MDS if mds is None else mds
        self.contexts: dict = getattr(self, 'contexts', {})

        if 'meta' in self.contexts:
            meta_dict = self.contexts['meta']
        else:
            meta_dict = self.contexts['meta'] = {}

        if meta:
            meta_dict.update(meta)

    def __call__(self, market_data: MarketData, **kwargs):
        # filter the out session data
        masked = self.mask_data(market_data=market_data, filter_mode=self.filter_mode)
        if not masked:
            return

        self.on_market_data(market_data=market_data, **kwargs)

        if isinstance(market_data, TickData):
            self.on_tick_data(tick_data=market_data, **kwargs)
        elif isinstance(market_data, (TradeData, TransactionData)):
            self.on_trade_data(trade_data=market_data, **kwargs)
        elif isinstance(market_data, OrderBook):
            self.on_order_book(order_book=market_data, **kwargs)
        elif isinstance(market_data, OrderData):
            self.on_order_data(order_data=market_data, **kwargs)
        else:
            raise NotImplementedError(f"Can not handle market data type {type(market_data)}")

    @classmethod
    def mask_data(cls, market_data: MarketData, filter_mode: FilterMode = FilterMode(7)) -> bool:
        if FilterMode.no_cancel in filter_mode:
            if isinstance(market_data, (TradeData, TransactionData)) and market_data.side == TransactionSide.UNKNOWN:
                return False

        if filter_mode.no_auction in filter_mode:
            if not GlobalStatics.PROFILE.is_market_session(timestamp=market_data.timestamp):
                return False

        if filter_mode.no_order in filter_mode:
            if isinstance(market_data, OrderData):
                return False

        if filter_mode.no_trade in filter_mode:
            if isinstance(market_data, (TradeData, TransactionData)):
                return False

        if filter_mode.no_tick in filter_mode:
            if isinstance(market_data, TickData):
                return False

        return True

    def on_market_data(self, market_data: MarketData, **kwargs):
        pass

    def on_tick_data(self, tick_data: TickData, **kwargs) -> None:
        pass

    def on_trade_data(self, trade_data: TradeData | TransactionData, **kwargs) -> None:
        pass

    def on_order_book(self, order_book: OrderBook, **kwargs) -> None:
        pass

    def on_order_data(self, order_data: OrderData, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def factor_names(self, subscription: list[str]) -> list[str]:
        """
        This method returns a list of string, corresponding with the keys of the what .value returns.
        This method is design to facilitate facter caching functions.
        """
        ...

    @classmethod
    def _params_grid(cls, param_range: dict[str, list[...]], param_static: dict[str, ...] = None, auto_naming: bool = None) -> list[dict[str, ...]]:
        """
        convert param grid to list of params
        Args:
            param_range: parameter range, e.g. dict(sampling_interval=[5, 15, 60], sample_size=[10, 20, 30])
            param_static: static parameter value, e.g. dict(weights=self.weights), this CAN OVERRIDE param_range

        Returns: parameter list

        """
        param_grid: list[dict] = []

        for name in param_range:
            _param_range = param_range[name]
            extended_param_grid = []

            for value in _param_range:
                if param_grid:
                    for _ in param_grid:
                        _ = _.copy()
                        _[name] = value
                        extended_param_grid.append(_)
                else:
                    extended_param_grid.append({name: value})

            param_grid.clear()
            param_grid.extend(extended_param_grid)

        if param_static:
            for _ in param_grid:
                _.update(param_static)

        if (auto_naming
                or ('name' not in param_range
                    and 'name' not in param_static
                    and auto_naming is None)):
            for i, param_dict in enumerate(param_grid):
                param_dict['name'] = f'Monitor.Grid.{cls.__name__}.{i}'

        return param_grid

    def collect_meta_info(self, **kwargs) -> dict[str, str | float | int | bool]:
        meta = self.contexts.get('meta', {})

        meta.update(
            name=self.name,
            type=self.__class__.__name__
        )

        meta.update(**kwargs)

        if isinstance(self, FixedIntervalSampler):
            meta.update(
                sampling_interval=self.sampling_interval,
                sample_size=self.sample_size
            )

        if isinstance(self, FixedVolumeIntervalSampler):
            # no additional meta info
            pass

        if isinstance(self, AdaptiveVolumeIntervalSampler):
            meta.update(
                baseline_window=self.baseline_window,
                aligned_interval=self.aligned_interval
            )

        if isinstance(self, Synthetic):
            meta.update(
                index_name=self.weights.index_name
            )

        if isinstance(self, EMA):
            LOGGER.warning('Meta info for EMA may not be accurate due to precision limitation!')

            meta.update(
                alpha=self.alpha,
                window=self.window
            )

        return meta

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            name=self.name,
            monitor_id=self.monitor_id
            # enabled=self.enabled  # the enable flag will not be serialized, this affects the handling of mask in multiprocessing
        )

        if isinstance(self, FixedIntervalSampler):
            data_dict.update(
                FixedIntervalSampler.to_json(self=self, fmt='dict')
            )

        if isinstance(self, FixedVolumeIntervalSampler):
            data_dict.update(
                FixedVolumeIntervalSampler.to_json(self=self, fmt='dict')
            )

        if isinstance(self, AdaptiveVolumeIntervalSampler):
            data_dict.update(
                AdaptiveVolumeIntervalSampler.to_json(self=self, fmt='dict')
            )

        if isinstance(self, VolumeProfileSampler):
            data_dict.update(
                VolumeProfileSampler.to_json(self=self, fmt='dict')
            )

        if isinstance(self, Synthetic):
            data_dict.update(
                Synthetic.to_json(self=self, fmt='dict')
            )

        if isinstance(self, EMA):
            data_dict.update(
                EMA.to_json(self=self, fmt='dict')
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

        var_names = cls.__init__.__code__.co_varnames
        kwargs = {name: json_dict[name] for name in var_names if name in json_dict}

        self = cls(**kwargs)

        self.update_from_json(json_dict=json_dict)

        return self

    def to_shm(self, name: str = None, manager: SharedMemoryCore = None) -> str:
        if name is None:
            name = f'{self.monitor_id}.json'

        if manager is None:
            from ..base.memory_core import SharedMemoryCore
            manager = SharedMemoryCore()

        serialized = pickle.dumps(self)
        size = len(serialized)

        manager.init_buffer(name=name, buffer_size=size, init_value=serialized)
        # shm.close()
        return name

    @classmethod
    def from_shm(cls, name: str = None, monitor_id: str = None, manager: SharedMemoryCore = None) -> Self:
        if name is None and monitor_id is None:
            raise ValueError('Must assign a name or monitor_id.')
        if name is None:
            name = f'{monitor_id}.json'

        if manager is None:
            manager = SharedMemoryCore()

        shm = manager.get_buffer(name=name)
        self: Self = pickle.loads(bytes(shm.buffer))

        return self

    def clear(self) -> None:
        if isinstance(self, EMA):
            EMA.clear(self=self)

        if isinstance(self, Synthetic):
            Synthetic.clear(self=self)

        if isinstance(self, FixedIntervalSampler):
            FixedIntervalSampler.clear(self=self)

        if isinstance(self, FixedVolumeIntervalSampler):
            FixedVolumeIntervalSampler.clear(self=self)

        if isinstance(self, AdaptiveVolumeIntervalSampler):
            AdaptiveVolumeIntervalSampler.clear(self=self)

    def update_from_json(self, json_dict: dict) -> Self:
        """
        a utility function for .from_json()

        Note that calling this function DOES NOT CLEAR the object
        This function will add /update / override the data from json dict

        Call .clear() explicitly if a re-construct is needed.
        """

        if isinstance(self, EMA):
            self: EMA
            for name in json_dict['ema']:
                self.register_ema(name=name)

                self._ema_memory[name].update(json_dict['ema_memory'][name])
                self._ema_current[name].update(json_dict['ema_current'][name])
                self.ema[name].update(json_dict['ema'][name])

        if isinstance(self, Synthetic):
            self: Synthetic

            self.base_price.update(json_dict['base_price'])
            self.last_price.update(json_dict['last_price'])
            self.synthetic_base_price = json_dict['synthetic_base_price']

        if isinstance(self, FixedIntervalSampler):
            self: FixedIntervalSampler

            for topic, sampler_json in json_dict['sampler_data'].items():
                sampler_data = SamplerData.from_json(sampler_json)
                self.sampler_data[topic] = sampler_data

        if isinstance(self, FixedVolumeIntervalSampler):
            self: FixedVolumeIntervalSampler

            self.contexts['vol_acc'].update(json_dict['vol_acc'])
            self.contexts['use_notional'] = json_dict['use_notional']

        if isinstance(self, AdaptiveVolumeIntervalSampler):
            self: AdaptiveVolumeIntervalSampler

            volume_baseline = self.contexts['vol_baseline']
            volume_baseline['baseline'].update(json_dict['volume_baseline']['baseline'])
            volume_baseline['sampling_interval'].update(json_dict['volume_baseline']['sampling_interval'])
            volume_baseline['obs_vol_acc_start'].update(json_dict['volume_baseline']['obs_vol_acc_start'])
            volume_baseline['obs_index'].update(json_dict['volume_baseline']['obs_index'])
            for ticker, data in json_dict['volume_baseline']['obs_vol_acc'].items():
                if ticker in volume_baseline['obs_vol_acc']:
                    volume_baseline['obs_vol_acc'][ticker].extend(data)
                else:
                    volume_baseline['obs_vol_acc'][ticker] = deque(data, maxlen=self.baseline_window)

        if isinstance(self, VolumeProfileSampler):
            self: VolumeProfileSampler
            from .volume_profile import VolumeProfile

            self.contexts['estimated_volume_interval'].update(json_dict['estimated_volume_interval'])
            self.volume_profile.update({ticker: VolumeProfile.from_json(profile_dict) for ticker, profile_dict in json_dict['volume_profile'].items()})

        return self

    def _param_range(self) -> dict[str, list[...]]:
        # give some simple parameter range
        params_range = {}
        if isinstance(self, FixedIntervalSampler):
            params_range.update(
                sampling_interval=[5, 15, 60],
                sample_size=[10, 20, 30]
            )

        if isinstance(self, AdaptiveVolumeIntervalSampler):
            params_range.update(
                aligned_interval=[True, False]
            )

        if isinstance(self, EMA):
            params_range.update(
                alpha=[ALPHA_05, ALPHA_02, ALPHA_01, ALPHA_001, ALPHA_0001]
            )

        return params_range

    def _param_static(self) -> dict[str, ...]:
        param_static = {}

        if isinstance(self, AdaptiveVolumeIntervalSampler):
            param_static.update(
                baseline_window=self.baseline_window
            )

        if isinstance(self, Synthetic):
            param_static.update(
                weights=self.weights
            )

        return param_static

    def params_list(self) -> list[dict[str, ...]]:
        """
        This method is designed to facilitate the grid cv process.
        The method will return a list of params to initialize monitor.
        e.g.
        > params_list = self.params_grid()
        > monitors = [SomeMonitor(**_) for _ in params_list]
        > # cross validation process ...
        Returns: a list of dict[str, ...]

        """

        params_list = self._params_grid(
            param_range=self._param_range(),
            param_static=self._param_static()
        )

        return params_list

    @property
    def params(self) -> dict:
        params = dict()

        signature = inspect.signature(self.__init__)
        for name, param in signature.parameters.items():
            if hasattr(self, name):
                params[name] = getattr(self, name)
            elif name in self.contexts:
                param[name] = self.contexts[name]
            elif param.default is not inspect.Parameter.empty:
                pass
            elif name == 'kwargs':
                LOGGER.warning(f'Kwargs detected! Can not determine the necessary parameters for {self.__class__.__name__}.')
            else:
                raise KeyError(f'Can not find required parameter {name} in {self}!')

        return params

    @property
    def serializable(self) -> bool:
        return True

    @property
    def use_shm(self) -> bool:
        return False

    @property
    def meta(self) -> dict[str, str | float | int | bool]:
        meta_info = self.collect_meta_info()
        return {k: meta_info[k] for k in sorted(meta_info)}

    def digest(self, encoding: str = 'utf-8') -> str:
        hashed_str = hashlib.sha256(json.dumps(self.meta, sort_keys=True).encode(encoding=encoding)).hexdigest()
        return hashed_str


class EMA(object):
    """
    Use EMA module with samplers to get best results
    """

    def __init__(self, alpha: float = None, window: int = None):
        self.alpha = alpha if alpha else 2 / (window + 1)
        self.window = window if window else round(2 / alpha - 1)

        if not (0 < alpha < 1):
            LOGGER.warning(f'{self.__class__.__name__} should have an alpha between 0 to 1')

        self._ema_memory: dict[str, dict[str, float]] = getattr(self, '_ema_memory', {})
        self._ema_current: dict[str, dict[str, float]] = getattr(self, '_ema_current', {})
        self.ema: dict[str, dict[str, float]] = getattr(self, 'ema', {})

    @classmethod
    def calculate_ema(cls, value: float, memory: float = None, window: int = None, alpha: float = None):
        if alpha is None and window is None:
            raise ValueError('Must assign value to alpha or window.')
        elif alpha is None:
            alpha = 2 / (window + 1)

        if memory is None:
            return value

        ema = alpha * value + (1 - alpha) * memory
        return ema

    def register_ema(self, name: str) -> dict[str, float]:
        if name in self.ema:
            LOGGER.warning(f'name {name} already registered in {self.__class__.__name__}!')
            return self.ema[name]

        self._ema_memory[name] = {}
        self._ema_current[name] = {}
        self.ema[name] = {}

        return self.ema[name]

    def update_ema(self, ticker: str, replace_na: float | dict[str, float] = np.nan, **update_data: float):
        """
        update ema on call

        Args:
            ticker: the ticker of the
            replace_na: replace the memory value with the gaven if it is nan
            **update_data: {'ema_a': 1, 'ema_b': 2}

        Returns: None

        """
        # update the current
        for entry_name, value in update_data.items():
            if entry_name not in self._ema_current:
                LOGGER.warning(f'Entry {entry_name} not registered')
                continue

            if not np.isfinite(value):
                LOGGER.warning(f'Value for {entry_name} not valid, expect float, got {value}, ignored to prevent data-contamination.')
                continue

            current = self._ema_current[entry_name][ticker] = value
            memory = self._ema_memory[entry_name].get(ticker, np.nan)

            if np.isfinite(memory):
                self.ema[entry_name][ticker] = self.calculate_ema(value=current, memory=memory, alpha=self.alpha)
            elif isinstance(replace_na, (float, int)):
                self.ema[entry_name][ticker] = self.calculate_ema(value=current, memory=replace_na, alpha=self.alpha)
            elif isinstance(replace_na, dict):
                self.ema[entry_name][ticker] = self.calculate_ema(value=current, memory=replace_na[entry_name], alpha=self.alpha)
            else:
                raise TypeError(f'Invalid {replace_na=}, expect float or dict of float.')

    def accumulate_ema(self, ticker: str, replace_na: float | dict[str, float] = np.nan, **accumulative_data: float):
        # add to current
        for entry_name, value in accumulative_data.items():
            if entry_name not in self._ema_current:
                LOGGER.warning(f'Entry {entry_name} not registered')
                continue

            if not np.isfinite(value):
                LOGGER.warning(f'Value for {entry_name} not valid, expect float, got {value}, ignored to prevent data-contamination.')
                continue

            current = self._ema_current[entry_name][ticker] = self._ema_current[entry_name].get(ticker, 0.) + value
            memory = self._ema_memory[entry_name].get(ticker, np.nan)

            if np.isfinite(memory):
                self.ema[entry_name][ticker] = self.calculate_ema(value=current, memory=memory, alpha=self.alpha)
            elif isinstance(replace_na, (float, int)):
                self.ema[entry_name][ticker] = self.calculate_ema(value=current, memory=replace_na, alpha=self.alpha)
            elif isinstance(replace_na, dict):
                self.ema[entry_name][ticker] = self.calculate_ema(value=current, memory=replace_na[entry_name], alpha=self.alpha)
            else:
                raise TypeError(f'Invalid {replace_na=}, expect float or dict of float.')

    def enroll(self, ticker: str, name: str):
        current = self.ema[name][ticker]
        latest = self._ema_current[name].get(ticker, np.nan)

        if np.isfinite(current):
            self._ema_memory[name][ticker] = current
        elif np.isfinite(latest):
            self._ema_memory[name][ticker] = latest

        self._ema_current[name].pop(ticker, None)

    def enroll_all(self):
        for name in self.ema:
            for ticker in self._ema_current:
                self.enroll(ticker=ticker, name=name)

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            alpha=self.alpha,
            window=self.window,
            ema_memory=self._ema_memory,
            ema_current=self._ema_current,
            ema=self.ema
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> EMA:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            alpha=json_dict['alpha'],
            window=json_dict['window']
        )

        for name in json_dict['ema']:
            self.register_ema(name=name)

            self._ema_memory[name].update(json_dict['ema_memory'][name])
            self._ema_current[name].update(json_dict['ema_current'][name])
            self.ema[name].update(json_dict['ema'][name])

        return self

    def clear(self):
        self._ema_memory.clear()
        self._ema_current.clear()
        self.ema.clear()


class Synthetic(object, metaclass=abc.ABCMeta):
    def __init__(self, weights: dict[str, float]):
        self.weights: IndexWeight = weights if isinstance(weights, IndexWeight) else IndexWeight(index_name='synthetic', **weights)
        self.weights.normalize()

        self.base_price: dict[str, float] = {}
        self.last_price: dict[str, float] = {}
        self.synthetic_base_price: float = 1.

    def composite(self, values: dict[str, float], replace_na: float = 0.):
        return self.weights.composite(values=values, replace_na=replace_na)

    def update_synthetic(self, ticker: str, market_price: float):
        if ticker not in self.weights:
            return

        base_price = self.base_price.get(ticker, np.nan)
        if not np.isfinite(base_price):
            self.base_price[ticker] = market_price

        self.last_price[ticker] = market_price

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            index_name=self.weights.index_name,
            weights=dict(self.weights),
            base_price=self.base_price,
            last_price=self.last_price,
            synthetic_base_price=self.synthetic_base_price
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> Synthetic:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            weights=IndexWeight(index_name=json_dict['index_name'], **json_dict['weights'])
        )

        self.base_price.update(json_dict['base_price'])
        self.last_price.update(json_dict['last_price'])
        self.synthetic_base_price = json_dict['synthetic_base_price']

        return self

    def clear(self):
        self.base_price.clear()
        self.last_price.clear()
        self.synthetic_base_price = 1.

    @property
    def synthetic_index(self):
        price_list = []
        weight_list = []

        for ticker, weight in self.weights.items():
            last_price = self.last_price.get(ticker, np.nan)
            base_price = self.base_price.get(ticker, np.nan)

            assert weight > 0, f'Weight of {ticker} in {self.weights.index_name} must be greater than zero.'
            weight_list.append(weight)

            if np.isfinite(last_price) and np.isfinite(base_price):
                price_list.append(last_price / base_price)
            else:
                price_list.append(1.)

        if sum(weight_list):
            synthetic_index = np.average(price_list, weights=weight_list) * self.synthetic_base_price
        else:
            synthetic_index = 1.

        return synthetic_index

    @property
    def composited_index(self) -> float:
        return self.composite(self.last_price)


__all__ = ['FactorInfo', 'FactorTree', 'FactorMonitor',
           'EMA', 'ALPHA_05', 'ALPHA_02', 'ALPHA_01', 'ALPHA_001', 'ALPHA_0001',
           'Synthetic', 'IndexWeight', 'FilterMode']
