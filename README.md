Quark is a python HFT trading / testing platform.

Quark is designed as a factor driven trading platform, which
- Can design, build, test, validate fit and ensemble the factors.
- Generates trade signal with the factor values.
- Handles market data and orders, providing position management.
- Ensures logic / code consistency between live-production and backtesting.
- Can be applied to trading stocks, stock indices, derivatives, etc.

---

# Structure
[![](https://mermaid.ink/img/pako:eNqlWG1vozgQ_isW0q66UltdCKlW0WmlFIi6SlG5gnorNf3ggptwJRAZotsoan_7jT2EAIEAOdQGjOeZ8bx4xsNO8WKfKWPlLYz_9ZaUp8S9nUcErmTzuuB0vSTm75TxiIbPc-Vr9Jqs58oLUpSoHIsAgROsiEVTb1kiEpcz2O3myoryd5YSn6Z0rnx8VEhUQZJy6jPC2TrmaVImYpE_j2pkm7-EbPM3KBAt2JFos1202SZaEg0PRDH3Ge-4PCeNOV0wsUadesvjBRq3zxdzZQ4vp9QDYiIe7TgOxYP4U769VMRU5OWy7mOPhuS0q-ztJFzEZrQIIlZPWmZJt4wPmgnFZRkOaAC_sNTjWcOawCz8Hs1WTHYsVz0tNye24igAy1k0AktzAcrekOxVLRyXPhDUGB3C8IaMkJfxeLwMfJ9FTTD1PNjwGV1dWR_6nMceSxLp-y68tLYlNAAHuIZyxInRII-4JmupTVC1FSo0v76-JvDfQTmtSVBUFFQPngoFC3vpiYYblrQLnapn4rT-uJrQF9eDAaweRHbp7NJ8E5S3tRgRHDb6xA8489IgjvK0X70mhVh5-nti741_wtGTQoy4HSHDA-Snx14ZX3RBaV0jqsHYtgUM7DgJpAlwJ65YlNZKfBTEj0HyXiWcR1--VIn1iX5nioKY5X6hGI18eRdFIIgW5RUfs3ia3AMDl9MgAuoSB4ivAAoZLLpZ7dbUOjydWp07S1QlBw4FzCcWW8V8i4XomHY66h__05szMN_7YfJSeao8_7WBDNqhDoqteB9zubfEMxGDhiRUtrR22tIyXgrbzIIjWdg5I-tqA7I1Ieu9ErKuNciJ2uSsQ-qxZRxCThsWDNF_uxqm_lypeEycX0Vxf2lzwqjdCbZwgg3hHngp8w8R1m4cWz0b2dEkklY7V0rBBdr_ckHZpjftNnULge1SvkDHdYpsV22Ctoa22yu0Xa1JUJ_YHvUwbNdEDfEdJJDm9X3eycZE3-eeKvtZwd4zFoZbLDk8gCYuoJ1sP1O7sDjpg1ln-8-0LsJaznyWoe9EXvgnBtyWVKz00d3eDnR3KVtsZS-bPddXx2ABLfHFhSzQ0A_ex4vAy-R9q2mBzAcDaM3YqG2QbuXsbcPsT1uHWfitb59O9YXGLfnz6uqH6M_whTMg5bEqx6I3yyBmhcJUkYUkwTa48KIiptSF4RwwAoZXos9CxgM5nMrbD3HMqNLhkiwV6dQaus_PT9FJ4X2Y6f8prjp-GvLTkJ9W4Jet3prIqQdDEhZO0WJoW_L2aFWV3Z8HMkPeWch_JMl11NJGnV0czQbZivQK5gYxqLGN-rs4mqnHGKFnXsPlyMabi7dZZpkjOd9RDlrCRru4OJppBzmIKpZtya9YRKovRjUYsBLJDrAkPzXAaQ0jztQzAoO9CVsfku-epMx-71U985_cgFlEgue-ik0Ev7BNSHX-0cpEObA19jNKiQwJvZAmCayHYLYib0EYXsVr6kFWGZM_LkmS8vidFV7lH2DSbcjyT2WdgAfYvknoibJ6AsxfPQH4MakfpvRxqac4_NB0BkjtBMIWC2GV70V94WfriM3GGSoOzwFp54BG54Buepswb6d6A4tnor7gvMB3ASqXyorxFQ18ZazshM5zJV1C4z9XxvDosze6CVM4vkQfQEo3aexsI08Zp3zDLpXNGlp0ZgQUjhar_Us4u4uww4_u8tv7x39nnIrr?type=png)](https://mermaid.live/edit#pako:eNqlWG1vozgQ_isW0q66UltdCKlW0WmlFIi6SlG5gnorNf3ggptwJRAZotsoan_7jT2EAIEAOdQGjOeZ8bx4xsNO8WKfKWPlLYz_9ZaUp8S9nUcErmTzuuB0vSTm75TxiIbPc-Vr9Jqs58oLUpSoHIsAgROsiEVTb1kiEpcz2O3myoryd5YSn6Z0rnx8VEhUQZJy6jPC2TrmaVImYpE_j2pkm7-EbPM3KBAt2JFos1202SZaEg0PRDH3Ge-4PCeNOV0wsUadesvjBRq3zxdzZQ4vp9QDYiIe7TgOxYP4U769VMRU5OWy7mOPhuS0q-ztJFzEZrQIIlZPWmZJt4wPmgnFZRkOaAC_sNTjWcOawCz8Hs1WTHYsVz0tNye24igAy1k0AktzAcrekOxVLRyXPhDUGB3C8IaMkJfxeLwMfJ9FTTD1PNjwGV1dWR_6nMceSxLp-y68tLYlNAAHuIZyxInRII-4JmupTVC1FSo0v76-JvDfQTmtSVBUFFQPngoFC3vpiYYblrQLnapn4rT-uJrQF9eDAaweRHbp7NJ8E5S3tRgRHDb6xA8489IgjvK0X70mhVh5-nti741_wtGTQoy4HSHDA-Snx14ZX3RBaV0jqsHYtgUM7DgJpAlwJ65YlNZKfBTEj0HyXiWcR1--VIn1iX5nioKY5X6hGI18eRdFIIgW5RUfs3ia3AMDl9MgAuoSB4ivAAoZLLpZ7dbUOjydWp07S1QlBw4FzCcWW8V8i4XomHY66h__05szMN_7YfJSeao8_7WBDNqhDoqteB9zubfEMxGDhiRUtrR22tIyXgrbzIIjWdg5I-tqA7I1Ieu9ErKuNciJ2uSsQ-qxZRxCThsWDNF_uxqm_lypeEycX0Vxf2lzwqjdCbZwgg3hHngp8w8R1m4cWz0b2dEkklY7V0rBBdr_ckHZpjftNnULge1SvkDHdYpsV22Ctoa22yu0Xa1JUJ_YHvUwbNdEDfEdJJDm9X3eycZE3-eeKvtZwd4zFoZbLDk8gCYuoJ1sP1O7sDjpg1ln-8-0LsJaznyWoe9EXvgnBtyWVKz00d3eDnR3KVtsZS-bPddXx2ABLfHFhSzQ0A_ex4vAy-R9q2mBzAcDaM3YqG2QbuXsbcPsT1uHWfitb59O9YXGLfnz6uqH6M_whTMg5bEqx6I3yyBmhcJUkYUkwTa48KIiptSF4RwwAoZXos9CxgM5nMrbD3HMqNLhkiwV6dQaus_PT9FJ4X2Y6f8prjp-GvLTkJ9W4Jet3prIqQdDEhZO0WJoW_L2aFWV3Z8HMkPeWch_JMl11NJGnV0czQbZivQK5gYxqLGN-rs4mqnHGKFnXsPlyMabi7dZZpkjOd9RDlrCRru4OJppBzmIKpZtya9YRKovRjUYsBLJDrAkPzXAaQ0jztQzAoO9CVsfku-epMx-71U985_cgFlEgue-ik0Ev7BNSHX-0cpEObA19jNKiQwJvZAmCayHYLYib0EYXsVr6kFWGZM_LkmS8vidFV7lH2DSbcjyT2WdgAfYvknoibJ6AsxfPQH4MakfpvRxqac4_NB0BkjtBMIWC2GV70V94WfriM3GGSoOzwFp54BG54Buepswb6d6A4tnor7gvMB3ASqXyorxFQ18ZazshM5zJV1C4z9XxvDosze6CVM4vkQfQEo3aexsI08Zp3zDLpXNGlp0ZgQUjhar_Us4u4uww4_u8tv7x39nnIrr)

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
- add new algo in `AlgoEngine/Engine/AlgoEngine.py`
- add or amend execution logic at `AlgoEngine/Engine/TradeEngine.py`

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