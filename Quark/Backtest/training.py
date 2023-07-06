__package__ = 'Quark.Backtest'

import datetime
import os
import pathlib
import re

import numpy as np
import pandas as pd
import plotly.graph_objs

from ..Calibration.linear import LinearCore
from ..Strategy.decoder import Decoder, RecursiveDecoder
from . import LOGGER

DATA_SOURCE = r'C:\Users\Bolun\Projects\Quark\TestResult.51bec829'
DATA_CORE = LinearCore(ticker='Synthetic')


def main():
    pattern = r'metric\.(\d{4}-\d{2}-\d{2})\.csv'

    for _ in os.listdir(DATA_SOURCE):
        if re.match(pattern, _):
            file_path = pathlib.Path(DATA_SOURCE, _)
            info = DATA_CORE.load_info_from_csv(file_path=file_path)

            report = DATA_CORE.calibrate(info=info)
            LOGGER.info(f'{DATA_SOURCE} calibration report:\n' + '\n'.join([f"{_}: {report[_]}" for _ in report]))
            fig = DATA_CORE.plot(info=info, decoder=DATA_CORE.decoder)
            fig.show()


if __name__ == '__main__':
    main()
