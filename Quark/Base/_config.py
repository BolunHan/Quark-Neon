import configparser
import json
import os.path
import pathlib
from collections import defaultdict
from types import SimpleNamespace

import dateutil
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
    config_dict = defaultdict(dict)

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
                sub_dict = sub_dict[prefix]

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
                    value = dateutil.parser(value)
                except Exception as _:
                    pass

                # noinspection PyBroadException
                try:
                    value = pd.to_numeric(value).item()
                except Exception as _:
                    value = value

            sub_dict[key] = value

    update_config(context=config_dict, config=CONFIG)


if os.path.isfile(CWD.joinpath('config.ini')):
    from_ini(CWD.joinpath('config.ini'))
else:
    raise FileNotFoundError(f'{CWD.joinpath("config.ini")} not found!')

# update some entries in GS
GlobalStatics.DEBUG_MODE = CONFIG.Telemetric.DEBUG_MODE
