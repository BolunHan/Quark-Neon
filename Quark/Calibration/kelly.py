import numpy as np


def kelly_bootstrap(outcomes: list[float], probs: list[float] = None, max_leverage: float = 2., adjust_factor: float = 1, cost: float = 0.) -> float:
    """
    Calculate the percentage of assets to be invested using the Kelly Criterion
    based on a set of probabilities and corresponding outcomes. If probs is None,
    outcomes are assumed to be equally weighted.

    $$
    f = \frac{p}{a} - \frac{1 - p}{b} = \dfrac{p}{a}(1 - \frac{1}{WLP} \frac{1}{WLR})
    $$

    Parameters:
    - outcomes (List[float]): List of potential outcomes.
    - probs (Optional[List[float]]): List of probabilities for each outcome. If None,
      outcomes are assumed to be equally weighted.

    Returns:
    float: The percentage of assets to be invested.
    """
    n = len(outcomes)

    if probs is None:
        probs = [1 / n] * n  # Equal weighting if probs is None
    elif len(outcomes) != len(probs):
        raise ValueError("Length of outcomes and probs must be the same")

    up_prob = []
    up_outcome = []
    down_prob = []
    down_outcome = []
    neutral_prob = []
    neutral_outcome = []

    for outcome, prob in zip(outcomes, probs):
        if outcome > cost:
            up_prob.append(prob)
            up_outcome.append(outcome)
        elif outcome < -cost:
            down_prob.append(prob)
            down_outcome.append(outcome)
        else:
            neutral_prob.append(prob)
            neutral_outcome.append(outcome)

    # for long action
    if not up_prob:
        kelly_long = 0.
    elif not (down_outcome + neutral_outcome):
        kelly_long = max_leverage
    else:
        expected_gain = np.average(up_outcome, weights=up_prob) - cost
        expected_loss = np.average(down_outcome + neutral_outcome, weights=down_prob + neutral_prob) - cost
        gain_prob = np.sum(up_prob)
        loss_prob = np.sum(down_prob + neutral_prob)

        kelly_long = gain_prob / -expected_loss - loss_prob / expected_gain

    # for short action
    if not down_prob:
        kelly_short = 0.
    elif not (up_prob + neutral_outcome):
        kelly_short = -max_leverage
    else:
        expected_gain = -np.average(down_outcome, weights=down_prob) - cost
        expected_loss = -np.average(up_outcome + neutral_outcome, weights=up_prob + neutral_prob) - cost
        gain_prob = np.sum(down_prob)
        loss_prob = np.sum(up_prob + neutral_prob)

        kelly_short = -(gain_prob / -expected_loss - loss_prob / expected_gain)

    if kelly_short < 0 < kelly_long:
        if kelly_long + kelly_short > 0:
            kelly_percentage = kelly_long
        elif kelly_long + kelly_short < 0:
            kelly_percentage = kelly_short
        else:
            kelly_percentage = 0.
    elif kelly_long > 0:
        kelly_percentage = kelly_long
    elif kelly_short < 0:
        kelly_percentage = kelly_short
    else:
        kelly_percentage = 0.

    kelly_percentage = max(-max_leverage, min(max_leverage, kelly_percentage))

    if adjust_factor:
        kelly_percentage *= adjust_factor

    return kelly_percentage
