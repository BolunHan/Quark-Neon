import configparser
import datetime
import json
import os.path
import pathlib
import traceback
import warnings
from typing import Self, overload, Any

import pandas as pd

from . import GlobalStatics


class ConfigDict(dict):
    _fields_ = ['name', 'parent', 'address']

    def __init__(self, name: str = 'root', **kwargs):
        super().__setattr__('name', name)
        super().__setattr__('parent', None)
        super().__setattr__('address', None)
        super().__init__()
        self.update_config(contents=kwargs, config=self)

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name: str, value):
        if name in self._fields_:
            raise KeyError(f'Can not assign name {name}! This keyword is reserved. Try use capitalized "{name.capitalize()}" instead!')

        self[name] = value

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self.name}{f' {self.address=}' if self.address else ''}>({", ".join([f'{name}: {f'Sub-{value.__class__.__name__}(...)' if isinstance(value, self.__class__) else value}' for name, value in self.items()])})'

    @classmethod
    def update_config(cls, contents: dict, config: Self) -> Self:

        for name, value in contents.items():
            if isinstance(value, dict):
                sub_config = cls()
                super(ConfigDict, sub_config).__setattr__('name', name)
                super(ConfigDict, sub_config).__setattr__('parent', config)
                super(ConfigDict, sub_config).__setattr__('address', name if config.address is None else f'{config.address}.{name}')

                cls.update_config(contents=value, config=sub_config)
                config.__setattr__(name=name, value=sub_config)
            elif '.' in name:
                raise KeyError(f'Invalid entry name {name}! Names should not contain dots!')
            else:
                config.__setattr__(name=name, value=value)
        return config

    @overload
    def get_config(self, *sub_address: str):
        ...

    @overload
    def get_config(self, *sub_address: str, default: Any):
        ...

    def get_config(self, *args, **kwargs):
        address_list = [name for _address in args for name in _address.split(".")]

        cfg = self
        for i, _addr in enumerate(address_list):
            if _addr not in cfg:
                if 'default' in kwargs:
                    return kwargs['default']

                raise KeyError(f'Can not find address {f"{cfg.address}.{_addr}" if cfg.address else _addr} in config {self}.')
            cfg = cfg[_addr]

        return cfg

    def set_config(self, *sub_address: str, value: Any):
        address_list = [name for _address in sub_address for name in _address.split(".")]

        cfg = self
        for i, _addr in enumerate(address_list[:-1]):
            if _addr in cfg:
                cfg = cfg[_addr]
            elif _addr not in cfg:
                sub_config = self.__class__()
                super(ConfigDict, sub_config).__setattr__('name', _addr)
                super(ConfigDict, sub_config).__setattr__('parent', cfg)
                super(ConfigDict, sub_config).__setattr__('address', _addr if cfg.address is None else f'{cfg.address}.{_addr}')
                cfg[_addr] = sub_config
                cfg = cfg[_addr]

        cfg[address_list[-1]] = value
        return cfg

    def tree(self, data=None, prefix="", parent_path="", full_path=False):
        """
        Recursively builds a tree-like string representation of a nested dictionary.

        :param data: The nested dictionary to display.
        :param prefix: The prefix used for the tree structure.
        :param parent_path: The accumulated path of the current node.
        :param full_path: If True, renders the full path of each node.
        :return: A string representation of the tree.
        """
        tree_string = ""

        if data is None:
            data = self

        if isinstance(data, dict):
            keys = list(data.keys())
            for i, key in enumerate(keys):
                is_last = i == len(keys) - 1
                connector = "└── " if is_last else "├── "
                current_path = f"{parent_path}.{key}" if parent_path else key
                value = data[key]
                if isinstance(value, (dict, list)):
                    display_key = current_path if full_path else key
                    tree_string += f"{prefix}{connector}{display_key}\n"
                    sub_prefix = "    " if is_last else "│   "
                    tree_string += self.tree(value, prefix + sub_prefix, current_path, full_path)
                else:
                    display_key = current_path if full_path else key
                    tree_string += f"{prefix}{connector}{display_key}: {value}\n"
        elif isinstance(data, list):
            for i, item in enumerate(data):
                is_last = i == len(data) - 1
                connector = "└── " if is_last else "├── "
                tree_string += f"{prefix}{connector}{item}\n"
                if isinstance(item, (dict, list)):
                    sub_prefix = "    " if is_last else "│   "
                    tree_string += self.tree(item, prefix + sub_prefix, parent_path, full_path)

        return tree_string

    def to_dict(self) -> dict[str, Any]:
        config_dict = {}
        for key, value in self.items():
            if isinstance(value, self.__class__):
                config_dict[key] = value.to_dict()
            else:
                config_dict[key] = value
        return config_dict


CONFIG = ConfigDict()
CWD = pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value)


def from_ini(file_path):
    config_dict = {}

    parser = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation(), allow_no_value=True)
    parser.optionxform = str
    parser.read(file_path)

    for section_name in parser.sections():
        section = parser[section_name]
        for key in section:
            value = section[key]
            address = section_name.split('.') + key.split('.')
            key = address[-1]

            sub_dict = config_dict
            for prefix in address[:-1]:
                if prefix not in sub_dict:
                    sub_dict[prefix] = {}

                sub_dict = sub_dict[prefix]
                assert isinstance(sub_dict, dict), f'Config key collision, {".".join(address)} assigned value to {sub_dict}!'

            if isinstance(value, str):

                # noinspection PyBroadException
                try:
                    # noinspection PyUnresolvedReferences
                    value = json.loads(value)
                except Exception as _:
                    pass

                # noinspection PyBroadException
                try:
                    # noinspection PyUnresolvedReferences
                    value = datetime.date.fromisoformat(value)
                except Exception as _:
                    pass

                # noinspection PyBroadException
                try:
                    value = pd.to_numeric(value).item()
                except Exception as _:
                    value = value

            if key not in sub_dict:
                sub_dict[key] = value
            else:
                assert not isinstance(sub_dict[key], dict), f'Config key collision, {".".join(address)} assigned as section {sub_dict[key]}!'
                sub_dict[key] = value

    CONFIG.update_config(contents=config_dict, config=CONFIG)


_configs = []
for file in os.listdir(CWD):
    if file.endswith(('.ini', '.INI')):
        try:
            from_ini(CWD.joinpath(file))
            _configs.append(file)
        except Exception as _:
            warnings.warn(f'Invalid configuration file {file}!\n{traceback.format_exc()}')

if not _configs:
    warnings.warn(f'No configuration file found at {CWD}!')

# update some entries in GS
GlobalStatics.DEBUG_MODE = CONFIG.get_config('Telemetric.DEBUG_MODE', default=False)
GlobalStatics.CONFIG = CONFIG
