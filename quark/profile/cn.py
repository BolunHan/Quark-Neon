import datetime

from algo_engine.profile import PROFILE_CN

from ..base import GlobalStatics


def profile_cn_override():
    """
    Override the global profile with a specific configuration for the Chinese market.

    This function updates the global settings for `RANGE_BREAK`, `TIME_ZONE`, `DUMMIES_SESSION_OPENING`,
    and `DUMMIES_SESSION_CLOSING` to match the Chinese market profile.

    The session times are set as:
    - DUMMIES_SESSION_OPENING: 10:00 AM
    - DUMMIES_SESSION_CLOSING: 2:30 PM
    """
    from .. import profile

    PROFILE_CN.override_profile(GlobalStatics.PROFILE)

    profile.RANGE_BREAK = GlobalStatics.RANGE_BREAK
    profile.TIME_ZONE = GlobalStatics.TIME_ZONE
    profile.DUMMIES_SESSION_OPENING = datetime.time(10, 0)
    profile.DUMMIES_SESSION_CLOSING = datetime.time(14, 30)
