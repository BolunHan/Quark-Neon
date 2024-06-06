__package__ = 'quark.base'

import logging

from algo_engine.engine import EVENT_ENGINE

# step 0: define exceptions
from ._exceptions import Exceptions

# step 1: init GlobalStatics from console args
from ._statics import GlobalStatics

# step 2: load config defined in GlobalStatics
from ._config import CONFIG

# step 3: init loggers and profiler
from ._telemetries import LOGGER, PROFILER


def safe_exit(code: int = 0, *args, **kwargs):
    import _thread

    try:
        _thread.interrupt_main()
    except KeyboardInterrupt as _:
        LOGGER.info('Daemon threads interrupted!')

    force_exit(code, *args, **kwargs)


def force_exit(code: int = 0, *args, **kwargs):
    import os
    exit_status = [f'{code}']

    if args:
        exit_status.append(f'{args}')

    if kwargs:
        exit_status.append(f'{kwargs}')

    LOGGER.info(f'Quark exit with code {", ".join(exit_status)}')

    from ._telemetries import CWD, PROCESS_ID
    lock_file = CWD.joinpath(f'Quark.{PROCESS_ID}.lock') if PROCESS_ID else CWD.joinpath(f'Quark.lock')
    if os.path.isfile(lock_file):
        os.remove(lock_file)

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
        EVENT_ENGINE.stop()
    except RuntimeError:
        pass


set_logger(logger=LOGGER)

start()
