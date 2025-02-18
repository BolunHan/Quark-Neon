__package__ = 'factor.corr'

__all__ = ['EntropyVolumeProfileMonitor']

__meta__ = {
    'ver': '1.1.0',
    'name': 'EntropyVolumeProfileMonitor',
    'params': [
        dict(
            sampling_interval=5,
            sample_size=60,
            ignore_primary=False,
            pct_change=True,
            name='Monitor.Entropy.VolumeProfile.5'
        ),
        dict(
            sampling_interval=15,
            sample_size=60,
            ignore_primary=False,
            pct_change=True,
            name='Monitor.Entropy.VolumeProfile.15'
        ),
        dict(
            sampling_interval=30,
            sample_size=30,
            ignore_primary=False,
            pct_change=True,
            name='Monitor.Entropy.VolumeProfile.30'
        ),
        dict(
            sampling_interval=60,
            sample_size=15,
            ignore_primary=False,
            pct_change=True,
            name='Monitor.Entropy.VolumeProfile.60'
        ),
        dict(
            sampling_interval=150,
            sample_size=10,
            ignore_primary=False,
            pct_change=True,
            name='Monitor.Entropy.VolumeProfile.150'
        )
    ],
    'family': 'corr',
    'market': 'cn',
    'requirements': [
        'quark @ git+https://github.com/BolunHan/Quark.git#egg=Quark',  # this requirement is not necessary, only put there as a demo
        'numpy',  # not necessary too, since the env installed with numpy by default
        'PyAlgoEngine',  # also not necessary
    ],
    'external_data': [],  # topic for required external data
    'external_lib': [],  # other library, like c compiled lib file.
    'factor_type': 'basic',
    'activated_date': None,
    'deactivated_date': None,
    'dependencies': [],
    'author': 'BolunHan',
    'comments':
        "Factor monitoring entropy of the stock return. "
        "This factor designed to minimizing prediction interval. "
}

from .entropy import EntropyVolumeProfileMonitor
