__package__ = 'quark.base'

import inspect
import logging
import os

from algo_engine.engine import EVENT_ENGINE

# step 0: define exceptions
from ._exceptions import Exceptions

# step 1: init GlobalStatics from console args
from ._statics import GlobalStatics

# step 2: load config defined in GlobalStatics
from ._config import CONFIG, ConfigDict

# step 3: init loggers and profiler
from ._telemetries import LOGGER, PROFILER


def safe_exit(code: int = 0, *args, **kwargs):
    # Get the caller information
    caller_frame = inspect.stack()[1]
    caller_info = f'{caller_frame.function} in {caller_frame.filename}:{caller_frame.lineno}'

    LOGGER.info(f'Exiting with code {code} by {caller_info}...')
    stop()

    try:
        import _thread
        _thread.interrupt_main()
    except KeyboardInterrupt as _:
        LOGGER.info('Daemon threads interrupted!')

    from ._telemetries import CWD, INFO
    lock_file = CWD.joinpath(f'{INFO.program}.{INFO.run_id}.lock') if INFO.run_id else CWD.joinpath(f'{INFO.program}.lock')
    if os.path.isfile(lock_file):
        os.remove(lock_file)

    force_exit(code, *args, **kwargs)


def force_exit(code: int = 0, *args, **kwargs):
    exit_status = [f'{code}']

    if args:
        exit_status.append(f'{args}')

    if kwargs:
        exit_status.append(f'{kwargs}')

    LOGGER.info(f'Quark exit with code {", ".join(exit_status)}')

    # noinspection PyUnresolvedReferences, PyProtectedMember
    os._exit(code)


def set_logger(logger: logging.Logger):
    import algo_engine
    import event_engine

    algo_engine.set_logger(logger=logger)
    event_engine.set_logger(logger=logger)


def start():
    EVENT_ENGINE.start()


def stop():
    try:
        if EVENT_ENGINE.active:
            EVENT_ENGINE.stop()
    except RuntimeError:
        pass


set_logger(logger=LOGGER)

# start()

__all__ = [
    'EVENT_ENGINE', 'GlobalStatics', 'CONFIG', 'ConfigDict', 'LOGGER', 'PROFILER', 'safe_exit', 'force_exit'
]
