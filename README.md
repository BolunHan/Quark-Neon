Quark is a python HFT program

Quark using info collected from components, to generate trade instruction for index.

---

# Structure

## Data Processing

Quark collect transaction data stream of all the corresponding components of the index.
- The market data is fed into a MarketDataService.
- MarketDataService can have many registered MarketDataMonitor, to process the data.
- Market data processing follows a NO-COPY, No-ALTERATION principle

## Factor Generation

for each MarketDataMonitor, the strategy collect its signal (factor) on new market data.
- use .value property of the monitor to collect generated factor.
- the factor can be a float, a dict of float, or any other types.
- the collected factor is logged by a metric module, for future usage such as training, reviewing.

## Trade Decision

Based on the collected signal and current position / balance, the decision core is to make a trade decision (signal)

By default, the strategy is using a dummy core, so that no trade action can be triggered while still processing signals.

In production mode, the dummy core can be replaced by a real decision core, like LinearCore.

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
pip install plotly pytz baostock
```

## Run

```shell
env QUARK_CWD="Quark/runtime" python "Quark/__main__.py"
```

- the `$QUARK_CWD` is to set a runtime dir for quark program
- the __main__.py is the entry point of the production

## Development

To add new factor:
- add monitor to `Quark/Strategy/data_core.py` for factor generation
- add the parsing code to the `Quark/Strategy/metric` file, both `collect_factors` method and `log_factors` method
- update decision core in `Quark/Calibration`

To implement new decision core
- add new module in `Quark/Calibration`

To further implement new trading behavior or trading algoes,
- clone the [PyAlgoEngine](https://github.com/BolunHan/PyAlgoEngine.git)
- add new algo in `AlgoEngine/Engine/AlgoEngine.py`
- add / amend execution logic in `AlgoEngine/Engine/TradeEngine.py`

To init / amend backtest
- edit `Quark/Backtest/__main__.py`
- run the script, with cwd as `/home/bolun/Projects/Quark/`, to resolve relative import issues
- to amend sim-match, edit `SimMatch` in PyAlgoEngine `AlgoEngine/Engine/TradeEngine.py`

Debug and Profiling

The telemetry module `Quark/Base/_trelemetric.py` offer a profiling module `PROFILER`.
- enable the profiler by setting parameter `enable=true`
- hook the profiler to a method by adding the decorator `@PROFILER.profile` to the method
- hook the profiler to a class by adding the decorator `@PROFILER.profile_all` to the class
- the profiler will provide a report and dump it to cwd
- Disable the profiler in production mode to avoid latency issues

API

Implement new trade / market / historical api in `Quark/API` module.

---

# Known Issues

- [ ] C++ version EventEngine has some performance issue.
- [ ] multi-threading is not tested.
- [ ] timezone awareness of datetime object is not supported. (due to an issue in `PyAlgoEngine`)