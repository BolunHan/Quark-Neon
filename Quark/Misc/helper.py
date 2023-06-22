import json
import os.path
import pathlib


def load_dict(file_path: str | pathlib.Path, json_str: str | bytes = None, json_dict: dict = None) -> dict[str, float]:
    if json_dict is None:
        json_dict = {}
    # this is for the shortcut using in backtest
    elif json_dict:
        return json_dict

    if json_str is None:
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                json_str = f.read()
        else:
            raise FileNotFoundError(f'{file_path} not exist')

    json_dict.update(json.loads(json_str))
    return json_dict
