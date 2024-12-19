import datetime
import pathlib
import time

import pandas as pd
from algo_engine.base import Progress
from algo_engine.profile import PROFILE_CN
from algo_engine.utils import fake_data


def run_backtester():
    from quark.apps.web_ui import WebApp, LOGGER

    PROFILE_CN.override_profile()

    ticker = '000016.SH'
    market_date = datetime.date.today()

    data_set = fake_data(market_date=market_date, interval=5)
    web_app = WebApp(start_date=market_date, end_date=market_date)
    LOGGER.info(f'{len(data_set)} fake data generated for {ticker} {market_date}.')

    web_app.register(ticker=ticker, interval=60)
    web_app.serve(blocking=False)

    LOGGER.info(f'web app started at {web_app.url}')

    for ts, row in Progress(list(data_set.iterrows())):
        web_app.update(
            timestamp=ts,
            ticker=ticker,
            open_price=row['open_price'],
            close_price=row['close_price'],
            high_price=row['high_price'],
            low_price=row['low_price'],
        )
        time.sleep(0.01)


def run_dashboard():
    from quark.apps.web_ui import WebApp

    PROFILE_CN.override_profile()

    here = pathlib.Path(__file__).parent.absolute()
    ticker = 'IH_MAIN'
    pred_var = ['pct_change_900', 'state_3', 'up_actual_3', 'down_actual_3']
    # pred_var = []
    factor = pd.read_csv(here.joinpath('factor.csv'), index_col=0)
    prediction = pd.read_csv(here.joinpath('prediction.csv'), index_col=0)
    data = pd.concat([factor, prediction], axis=1)
    data = data.dropna(subset=[f'{ticker}.market_price'])
    market_date = datetime.datetime.fromtimestamp(factor.index[0]).date()

    web_app = WebApp(start_date=market_date, end_date=market_date)
    web_app.register(ticker=ticker, interval=60, pred_var=pred_var)
    web_app.serve(blocking=False)
    current_pos = 0

    for ts, row in Progress(list(data.iterrows())):
        market_price = row[f'{ticker}.market_price']
        prediction = {}

        for _pred_var in pred_var:
            if _pred_var not in row:
                continue

            prediction[_pred_var] = row[_pred_var]
            prediction[f'{_pred_var}.upper_bound'] = row[f'{_pred_var}.upper_bound']
            prediction[f'{_pred_var}.lower_bound'] = row[f'{_pred_var}.lower_bound']

        if prediction['state_3.lower_bound'] > 0 and current_pos == 0:
            action = 1
        elif prediction['state_3.upper_bound'] < 0 and current_pos == 0:
            action = -1
        elif prediction['state_3'] > 0 and current_pos < 0:
            action = 1
        elif prediction['state_3'] < 0 and current_pos > 0:
            action = -1
        else:
            action = 0

        web_app.update(
            timestamp=ts,
            ticker=ticker,
            market_price=market_price,
            action=action,
            signal=row['markov'],
            **prediction
        )

        current_pos += action
        time.sleep(0.01)


if __name__ == '__main__':
    # run_backtester()
    run_dashboard()

    # sys.exit(-1)
