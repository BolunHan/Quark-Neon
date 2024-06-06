Quark is a python HFT trading / testing platform.

Quark is designed as a factor driven trading platform, which
- Can design, build, test, validate fit and ensemble the factors.
- Generates trade signal with the factor values.
- Handles market data and orders, providing position management.
- Ensures logic / code consistency between live-production and backtesting.
- Can be applied to trading stocks, stock indices, derivatives, etc.

---

# Structure
[![](https://mermaid.ink/img/pako:eNqlWG1vozgQ_isW0q66UltdCKlW0WmlFIi6SlG5gnorlX5wwSVcCY4M0W0Utb_9xjZJgEB4OdQGjOeZGc-MZzzsFJ8GRJkqbzH9119iliH31ksQXOnmNWR4vUTm74ywBMfPnvI1eU3XnvIiKUpUjgXzTrRCFs78ZYmGX85ot_OUFWbvJEMBzrCnfHxUSFROkjEcEMTImrIsLRORJPCSGtHmLxBt_gb1k5CcSDbbJZttkgXR-EhEWUBYR-2cjDIcElBRx_7yVD_j9vnCUzx4Occ-0CL-aFMa8wf-p3x7qUipiDuIuqd-m5vs7SwOqZmEUUJqKcsM8ZawUSMdvyzDAfXhF_Q8nTWsGczC78lsxVynYtWzYg-0Fk0isJqFEzAyA0z-AuVvatFScb4wS8YFt7khYuNlOp0uoyAgSRNMHQYbP0svV_ST7mbUJ2kq3N6Fl9amQgNwJHUoBxsfjQ7B1mQttQmqtkL5yq-vrxH8d1ic1iQoKQqqB8_5Agvb6AnHG5K2C52rA3Faf1xN4PPrwQBWDzyvdHbpYQ-UtjQfIDlqdEkQMeJnEU0O2b56zQqh8vT3zN7b_oyfZ4UQcTtCxkfIT5-8EhZ2QWldA6rB1javVTZNI2ECuRFXJMlqJT5y4scofa8SesmXL1VifabfmbwQyqQvXImTQNx5-o-SsKzxKYun2T0wcBmOEqAucYDwiqCCgdLNy27Nq-OzedW5s3g5cuAoQAJkkRVlW1mBTmnnk_7RP78ZgPneD3OokefK8l8byJ_tFZDvw3vK-Mbij4g_NySgspm1s2YWsVLYYhacwuLOyVhXG5CtuVjvlYt1rUFO0iZnHWOfLGkclOOt_1Y1TP25XOxcRvLtYBB-eOUV_qXNG5NWb9jcGzYEfeRnJDjGWbuVbHUwsqNtBK02VErBF9r_8kXZpDetJnULAe5iFkoHdopwV22Ctoa42yvEXa1JUJ8Yn_Swa9dkDdEdpZDq9Tz75EOk7zNQlfuiYO4FieOtrDosgvYtwp1Mv1C7sDjrgkVn8y-0LsJaTn2Woe94eviHAm6LKlb66G5uBzq7jIRbUbzlY32BjELohS8uRImGVvCehpGfi_tW0wGZDwbQmtSo7Y9uxextw-xPW4dZ-K3vns71hMYt-vPq6gdvz-QLZ4TKY1WMeWuWQ8wKhalKFoJEdsCFFxUxpS5MzgEjYHjFGy3JeCSGc3H7wU8aVTqpkqVKOrWG7vPzk7dS8j7O1__Jrzp-muSnSX5agV-uvTUTUw-GIDweo8XQtsTt0aoudn8myA15Z0n-E0Guy1Xacs2uHC1GuUZ6BXMjMXLFtly_K0cL9RTD13mo5GJky5srb4vcMidyvks50hK2tIsrRwvtKEeiisVb8CtWkOqLSQ0GrITyIyw6nB3gwCYjztRzAoO8cVsfU--epMx-71U995_YgHlEgue-8k0Ev7BNUHX-0cpFOfzMkM8oJTJJ6Mc4TUEfJJMVeovi-IqusQ9JZYr-uERpxug7Kbw6fHzJtjE5fCPrBDzC9m1CT5TVE2D-6gkQH5J6YopflvqKE5-ZhoDUTiDZZElYOVP1hg9eo-g3hixxPASkDQFNhoBueptwnz77Awsnot7gfVHvBFQulRVhKxwFylTZ8TV7SraE1t9TpvAYkDe8iTM4vSQfQIo3GXW2ia9M33Cckktls4YunRgRhqPF6vAWju488OT3dvHZ_eM_0LSMxA?type=png)](https://mermaid.live/edit#pako:eNqlWG1vozgQ_isW0q66UltdCKlW0WmlFIi6SlG5gnorlX5wwSVcCY4M0W0Utb_9xjZJgEB4OdQGjOeZGc-MZzzsFJ8GRJkqbzH9119iliH31ksQXOnmNWR4vUTm74ywBMfPnvI1eU3XnvIiKUpUjgXzTrRCFs78ZYmGX85ot_OUFWbvJEMBzrCnfHxUSFROkjEcEMTImrIsLRORJPCSGtHmLxBt_gb1k5CcSDbbJZttkgXR-EhEWUBYR-2cjDIcElBRx_7yVD_j9vnCUzx4Occ-0CL-aFMa8wf-p3x7qUipiDuIuqd-m5vs7SwOqZmEUUJqKcsM8ZawUSMdvyzDAfXhF_Q8nTWsGczC78lsxVynYtWzYg-0Fk0isJqFEzAyA0z-AuVvatFScb4wS8YFt7khYuNlOp0uoyAgSRNMHQYbP0svV_ST7mbUJ2kq3N6Fl9amQgNwJHUoBxsfjQ7B1mQttQmqtkL5yq-vrxH8d1ic1iQoKQqqB8_5Agvb6AnHG5K2C52rA3Faf1xN4PPrwQBWDzyvdHbpYQ-UtjQfIDlqdEkQMeJnEU0O2b56zQqh8vT3zN7b_oyfZ4UQcTtCxkfIT5-8EhZ2QWldA6rB1javVTZNI2ECuRFXJMlqJT5y4scofa8SesmXL1VifabfmbwQyqQvXImTQNx5-o-SsKzxKYun2T0wcBmOEqAucYDwiqCCgdLNy27Nq-OzedW5s3g5cuAoQAJkkRVlW1mBTmnnk_7RP78ZgPneD3OokefK8l8byJ_tFZDvw3vK-Mbij4g_NySgspm1s2YWsVLYYhacwuLOyVhXG5CtuVjvlYt1rUFO0iZnHWOfLGkclOOt_1Y1TP25XOxcRvLtYBB-eOUV_qXNG5NWb9jcGzYEfeRnJDjGWbuVbHUwsqNtBK02VErBF9r_8kXZpDetJnULAe5iFkoHdopwV22Ctoa42yvEXa1JUJ8Yn_Swa9dkDdEdpZDq9Tz75EOk7zNQlfuiYO4FieOtrDosgvYtwp1Mv1C7sDjrgkVn8y-0LsJaTn2Woe94eviHAm6LKlb66G5uBzq7jIRbUbzlY32BjELohS8uRImGVvCehpGfi_tW0wGZDwbQmtSo7Y9uxextw-xPW4dZ-K3vns71hMYt-vPq6gdvz-QLZ4TKY1WMeWuWQ8wKhalKFoJEdsCFFxUxpS5MzgEjYHjFGy3JeCSGc3H7wU8aVTqpkqVKOrWG7vPzk7dS8j7O1__Jrzp-muSnSX5agV-uvTUTUw-GIDweo8XQtsTt0aoudn8myA15Z0n-E0Guy1Xacs2uHC1GuUZ6BXMjMXLFtly_K0cL9RTD13mo5GJky5srb4vcMidyvks50hK2tIsrRwvtKEeiisVb8CtWkOqLSQ0GrITyIyw6nB3gwCYjztRzAoO8cVsfU--epMx-71U995_YgHlEgue-8k0Ev7BNUHX-0cpFOfzMkM8oJTJJ6Mc4TUEfJJMVeovi-IqusQ9JZYr-uERpxug7Kbw6fHzJtjE5fCPrBDzC9m1CT5TVE2D-6gkQH5J6YopflvqKE5-ZhoDUTiDZZElYOVP1hg9eo-g3hixxPASkDQFNhoBueptwnz77Awsnot7gfVHvBFQulRVhKxwFylTZ8TV7SraE1t9TpvAYkDe8iTM4vSQfQIo3GXW2ia9M33Cckktls4YunRgRhqPF6vAWju488OT3dvHZ_eM_0LSMxA)

## Data Processing

Quark collect lv2 data stream.
- The market data is fed into a MarketDataService (`MDS`).
- Multiple FactorMonitor are registered at `MDS`, each representing a factor.
- Utilizing multiprocessing and shared memory to achieve low latency and high performance.
- Market data processing follows a NO-COPY, No-ALTERATION principle.

## Factor Generation

for each MarketDataMonitor, the strategy collect its signal (factor values) on a given interval.
- Use .value property of the monitor to collect generated factor.
- the factor value must be a float, or a dict of float.
- A MonitorManager is provided to collect signal from other processes.
- Collected factor values are logged by a metric module, for decision-making and other future usage such as training, reviewing.

## Fitting

With the collected factor values, the Calibration module provides several fitting algos for different prediction targets.

By default, a general linear module with bootstrap is provided.

7 prediction targets are also provided, see [prediction target]().

## Trade Decision

Based on the collected signal and current position / balance, the decision core is to make a trade decision (signal).

By default, the strategy is using a dummy core, so that no trade action can be triggered without proper initialization.

In production mode, the dummy core can be Override by a real decision core, like a MajorityDecisionCore.

---

# Usage

## Installation

clone this model with 

```shell
git clone https://github.com/BolunHan/Quark.git
```

setup a python venv

```shell
python3 -m venv venv_quark
. venv_quark/bin/activate
```

install requirements, these packages are required for production
```shell
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Optional: install optional requirements, these package are required for backtesting 
```shell
pip install plotly pytz baostock, py7zr scikit-learn xgboost forestci statsmodels
```

## Run

### For Production

```shell
env QUARK_CWD="Quark/runtime" python "Quark/__main__.py"
```

- the `$QUARK_CWD` is to set a runtime dir for quark program
- the __main__.py is the entry point of the production

### For Backtesting

```shell
env QUARK_CWD="Quark/runtime" python "Quark/Backtest/__main__.py"
```

### For Factor Training and Validation

```shell
bash validation.sh
```

Note that this might take a lot more ram than production and backtest.

## Development

A variance of factor template is provided at `Quark/Factor/utils.py` and `Quark/Factor/utils_shm.py`.

To use these template, inheritance is advised. See the demo at `Quark/Factor/TradeFlow/trade_flow.py` for more details.

To add a new factor:
- Add your code at `Quark/Factor/<your factor dir>/<your factor scrtip>.py`.
- Import your script at `Quark/Factor/__init__.py`
- If the factor value need custom collection function, amend the function `collect_factor()` at `Quark/Factor/__init__.py`.
- Update validation script to train and test your factor. Add your factor at `initialize_factor` function, and the feature name at `self.features` list.
- For production and backtest, also update decision core in `Quark/DataLore/data_lore.py`. Add the feature name to `self.input_vars`.
- Also, if the factor does not follow the naming convention, update the `Quark/Factor/factor_pool.py` for proper caching.

To implement new data lore
- add new module in `Quark/DataLore`

To implement new decision core
- add new module in `Quark/DecisionCore`

To further implement new trading behavior and trading algorithm,
- clone the [PyAlgoEngine](https://github.com/BolunHan/PyAlgoEngine.git)
- add new algo in `algo_engine/engine/algo_engine.py`
- add or amend execution logic at `algo_engine/engine/trade_engine.py`

To init / amend backtest
- edit `Quark/Backtest/__main__.py`
- run the script, with cwd as `/home/bolun/Projects/Quark/`, to resolve the relative import.
- to amend sim-match logic, edit `SimMatch` in PyAlgoEngine `AlgoEngine/Engine/TradeEngine.py`

Debug and Profiling

A telemetry module `Quark/Base/_trelemetric.py` offer a profiling module `PROFILER`.
- enable the profiler by setting parameter `enable=true`
- hook the profiler to a method by adding the decorator `@PROFILER.profile` to the method
- hook the profiler to a class by adding the decorator `@PROFILER.profile_all` to the class
- the profiler will provide a report and dump it to cwd.
- Disable the profiler in production mode to avoid latency issues.

API

Implement new trade, market, historical api at `Quark/API` module.

---

# Known Issues and Roadmap

- [ ] C++ version EventEngine has some performance issue.
- [ ] Shared memory feature having performance issue (but still better than singleton). Resource tracker giving false alarm (a [python issue](https://bugs.python.org/issue39959)).
- [ ] IPC partially implemented.
- [ ] Timezone awareness of datetime object is not tested.