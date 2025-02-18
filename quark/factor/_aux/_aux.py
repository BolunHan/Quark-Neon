__all__ = ['TrendIndexVolumeProfileAuxiliaryMonitor']

__meta__ = {
    'ver': '1.0.0',
    'name': 'TrendIndexVolumeProfileAuxiliaryMonitor',
    'params': [
        dict(
            sampling_interval=5,
            sample_size=60,
            alpha=0.0739,  # which is 1 - ALPHA_001
            name='Monitor.Aux.Adaptive.Index.5'
        ),
        dict(
            sampling_interval=15,
            sample_size=60,
            alpha=0.0739,  # which is 1 - ALPHA_001
            name='Monitor.Aux.Adaptive.Index.15'
        ),
        dict(
            sampling_interval=30,
            sample_size=30,
            alpha=0.0739,  # which is 1 - ALPHA_001
            name='Monitor.Aux.Adaptive.Index.30'
        ),
        dict(
            sampling_interval=60,
            sample_size=15,
            alpha=0.0739,  # which is 1 - ALPHA_001
            name='Monitor.Aux.Adaptive.Index.60'
        )
    ],
    'family': 'low-pass',
    'market': 'cn',
    'requirements': [],
    'external_data': [],  # topic for required external data
    'external_lib': [],  # other library, like c compiled lib file.
    'factor_type': 'auxiliary',
    'activated_date': None,
    'deactivated_date': None,
    'dependencies': [],
    'comments': """
    An auxiliary factor for fitting factors having no predictive power on trend.
    """
}

from quark.factor.misc.auxiliary import TrendIndexVolumeProfileAuxiliaryMonitor
