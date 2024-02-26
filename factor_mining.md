# Factor Pool Building

In the context of this project, this article will briefly discuss the guidelines for building factor pools and factor mining principle.

# 1. Principle of Factors Design

## 1.1 Stationary

In high-frequency trading, the main difficulties and challenges are stamp duty and commission, in other words the frictional cost of trading. Therefore, in the process of factor design, the stationary of the factor comes first in the consideration.

Even in low and mid-frequency trading, the annualized return of a multi-factor quantitative market neutral strategy is in the range of 8% to 10%. With a daily turnover rate of about 10%, friction cost is an important factor that cannot be ignored. Therefore, in the trading of quantitative factors is designed to maintain the stability of the factor is extremely important.

The stability of the factor includes:
- Stable Expectations (Stationary)
- Limited Outliers (Magnitude and Amplitude)
- Auto-correlation (Stationary)
- a Stable Variance or a Stable Distribution

Several tools will be used to measure this:
- `statsmodels.tsa.stattools.adfuller`
- `statsmodels.tsa.stattools.kpss`
- Histogram distribution (manual test)
- `pd.plotting.autocorrelation_plot`

In this project, using any of the `validator` in `Quark/Factor/validation.py`, the factors will be plotted to facilitate manual testing. 

For simplicity, use following code snippet

```python
import datetime

from Quark.Factor.validation import FactorValidation

validator = FactorValidation(
    start_date=datetime.date(2023, 1, 1),
    end_date=datetime.date(2023, 2, 1)
)

validator.run()
```

Note that,
- the `initialize_factor` function must be amended accordingly before the validation starts.
- also `features` need to be updated with the `factor_name` of your factor.

After complete of the test, a directory with name like `FactorValidation.1` will be created, contains multiple subdirectory. In the subdirectory, plotting of the factor is generated as a `.html` file.

## 1.2 Prediction Target

The prediction targets for the factors are categorized into left and right sides.

The left-side targets include
- Return of Fixed (temporal) Interval 
- Return of Next Market Trend
- Return to Next Local Extreme

The right-side targets include
- Class of Current Trend 
- State of Markov Chain
- Return of Current Market Trend

In define the prediction targets, the project provides some utility functions:

These utility function takes market data and returns future info, thus are considered dangerous.

All future functions are located at `Quark/Factor/future.py` module.

Which has:
- Return of Fixed (temporal) Interval: `fix_prediction_target`
- Return of Market Trend: `wavelet_prediction_target`

The `wavelet_prediction_target` returns several prediction target together:
- `up_actual`, `down_actual`: the return to the next local minimal / maximal
- `target_actual`: aggregation of the two targets above
- `up_smoothed`, `down_smoothed`: smooth out so that prediction targets can be continuous.
- `target_smoothed`: also the aggregation
- `state`: sign of current trend.

The market trend is defined as a local extreme to the next local extreme. e.g. from a local minimal to next local maximal is an upward trend, marked as `state = 1`.

Local extreme can be decoded with arbitrary level. e.g. with `level = 2` use local minimal of the local minimals -> next local maximal of the local maximals. Therefore, the decoding process is recursive, which is the logic behind `RecursiveDecoder` at `Quark/Factor/decoder.py`.

To measure the performance of factors, this project uses all the 7 prediction targets we talk about above. And use the average metrics of them. This approach includes both left-side and right-side definition.

## 1.3 Fitting

The dominant fitting methods currently used are mainly linear and tree models. High-frequency data contains more noise than usual and is more prone to overfitting, and some models do not meet the task requirements.

Note: always be aware of the Bias-Variance tradeoff:

$$$
variance + bias^2 = MSE
$$$

Models such as random forests produce large variance making the fit less meaningful.

The goal of the fit should not just be the expectation (expected return), it should be the prediction interval (quantile loss function). Only then can there be an estimate of the variance and the predicted value be meaningful.

This project provides models:
- Linear Regression (ridge and ridge logistic) with bootstrap.
- Random Forest Estimator.
- A xgboost model

Linear model is used in the factor validation to achieve optimal performance.

Note that some weak factors were used in the model, so the linear model did not incorporate l1 regulation.

Some prediction targets have non-normal distribution, so kernel tricks (sigmoid and log) are used in those scenario.

New factor values are generated daily and added to the factor pool, and the model is fitted using a rolling window, with fix memory decay and exponential memory decay to mix in new fitted parameters. Long-term memory fades completely within 1 month. The decay rate is parameterized and can be adjusted at any time (to control for the effects of anomalies and faster style transitions).

Note that bootstrap produces biased forecasts, which have little effect on statistical arbitrage but unknown effects on CTP strategies. This can be corrected using unbiased estimator (like mean). For HFT, the bias impact is marginal (much less than the transaction costs).

The fitting models are located at `Quark/Calibration`. Using low level fitting method is not advised. DataLore is designed to manage the fitting and prediction process.

To use DataLore

```python
from Quark.DataLore.data_lore import LinearDataLore

data_lore = LinearDataLore(ticker='000016.SH', alpha=0.1, trade_cost=0.001)

# to predict
data_lore.predict(...)
# to calibrate
data_lore.calibrate(...)
```

The `Alpha` is the confidence level (both sides) of the prediction we use. `trade_cost` representing the percentage cost of each trade.

Again I recommend that DO NOT use or tweak the models, data lore and other related functions. Use `validation.py` instead.

## 1.4 Metrics

Since we adopt the bootstrap and bag design ideology, it is natural for us to use auc-roc as a metric.

Note that the problem of quantile crossing may accrue if unbiased estimation with bootstrap is used, as well as fitting with quantile loss (xgboost). Such time periods (which rarely occur) need to be filtered out when calculating metrics.

For example, when calculating the auc roc for accuracy.
- For a given `alpha`, if lower bound > trading cost then consider upwards pred.
- if upper bound < -trading cost then downwards pred.

Neutral prediction occurs when calculating auc roc, we also need to filter out those.

If there are too many neutral forecasts, it is also harmful for the strategy. In the `Quark/Factor/validation.py` module, auc curves and selection rate curves are plotted to assist us in manually calibrating factor performance.

Another metrics is the kelly criterion, which is a simple kelly formula that allows us to calculate the optimal leverage since we also give the distribution of expected returns.
- Positive leverage is upwards
- Negative leverage is downwards
- The kelly formula also gives a neutral prediction due to the trade costs.

The kelly prediction should not differ much from accuracy, which is essentially a different weighting of the predicted values, but if it does, then the kernel function and l2 regulation should be checked.

Other metrics, including `MSE`, `MSAE`, etc., are implemented in `Quark/Calibration/cross_validation.py`.

The mean of auc roc of accuracy is used in this project, calculated by cross validation. The EMA value of this metric is used in the optimal parameter selection.

## 1.5 Factor Ensemble

Since we need to predict both the value and boundary, and the prediction target contains both left-side and right-side targets. The introduction of certain factors which are not meaningful for trend prediction (called weak factors) sometimes can improve the metrics.

For example, the entropy factor can indicate that the market is about to reverse, but it is not possible to tell the direction of the market. The introduction of such a factor narrows the prediction interval of the left-side target. Using a certain l2 l1 regulation, together with a quadratic formula, it is possible to reduce the variance without significantly increasing the bias (because of the introduction of new information).

These weak factors are one of the reasons why this project can beat other similar algorithms.


If a tree model is used for fitting, factor screening is necessary:
- First, iterate all to find the factor that predicts best.
- Fitting the other factors and collect the residuals.
- The residuals are treated as new factors to fit the pred target.
- Repeat the steps above until the optimal AIC or BIC is achieved.
- In fact, after the optimal step, 2-3 more factors should be added (due to the unnecessarily large penalties from AIC BIC.)

This project uses a linear model, the multi co-linearity issue only affects the model interpretability, not the model prediction performance, so no residual fitting is performed.

# Building a Factor Pool

## 2.1 Calculate Factor Metrics

Factors should essentially be considered as operators. In this project, each factor corresponds to 15-18 parameter sets. 

Under these parameter sets, the optimal parameter set is selected 
- Calculate metrics of each set with CV algorithm.
- Use EMA of the metrics as the indicator.
- Select the param set with the best indicator, as the factor for next day.
- Calculate the performance of selected factor.
- This is considered as the **best possible outcome**.

If a factor's best possible outcome reaches the factor pool's threshold, then it can be considered to be added into the factor pool

## 2.2 Multi Co-linearity Issue

As in the previous parts [Factor Ensemble](), each time a factor is added, a residual screening is required to ensure that homogeneous factors are not submitted repeatedly.

In the early stages of building factor pool, this work is not very meaningful. Just check the codes manually.

Also note that decoration of existing factors (e.g., adding EMA, Diff, etc. to the operator) can be harmful to the factor pool (contaminating the weights for machine learning), and need to be screen out manually when checking the code.

## 2.3 Weak Factors

When doing residual screening, the weak factors need to be bound to a given strong factors.

When considering the addition of a weak factor, if no corresponding strong factor is given, we should iterate through the whole factor pool. 

To avoid over-fitting, only 1-to-1 pairs are considered in this process.

## 2.4 Human Data Mining

Factors should be developed using a limited dataset. Additional datasets (which can not be reached by the researchers) should be used to measure the performance of the factors.

The number of factor submissions needs to be limited (typically 3 times) to avoid human data mining issue.