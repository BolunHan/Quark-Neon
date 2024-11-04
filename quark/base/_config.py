import configparser
import datetime
import json
import os.path
import pathlib
import traceback
import warnings
from types import SimpleNamespace

import pandas as pd

from . import GlobalStatics

CONFIG = SimpleNamespace()
CWD = pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value)


def update_config(context: dict, config: SimpleNamespace) -> SimpleNamespace:
    for name, value in context.items():
        if isinstance(value, dict):
            setattr(config, name, update_config(value, SimpleNamespace()))
        else:
            setattr(config, name, value)
    return config


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

    update_config(context=config_dict, config=CONFIG)


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
try:
    GlobalStatics.DEBUG_MODE = CONFIG.Telemetric.DEBUG_MODE
except Exception as _:
    pass

GlobalStatics.CONFIG = CONFIG
