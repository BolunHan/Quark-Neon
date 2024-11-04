import functools
import glob
import inspect
import json
import logging
import logging.handlers
import os
import pathlib
import sys
import threading
import time
import traceback
import warnings
from collections import defaultdict
from enum import Enum
from types import SimpleNamespace

import pandas as pd
from algo_engine.base.telemetrics import ColoredFormatter

from . import GlobalStatics, CONFIG
from ._statics import get_current_path

FILE_PATH = get_current_path()
CWD = pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value)
try:
    LOG_LEVEL = CONFIG.Telemetric.LOG_LEVEL
except AttributeError:
    LOG_LEVEL = logging.INFO

INFO = SimpleNamespace(
    program='Quark',
    description='Telemetries tools for Quark and all its sub-modules',
    process_id=0,  # to support multiple process running at same time
    init_ts=time.time()
)

# step -1: instance a lock file
while True:
    PROCESS_ID = INFO.process_id
    lock_file = CWD.joinpath(f'Quark.{PROCESS_ID}.lock') if PROCESS_ID else CWD.joinpath(f'Quark.lock')

    if not os.path.isfile(lock_file):
        with open(lock_file, 'w') as f:
            f.write(json.dumps(INFO.__dict__))

        break

    INFO.process_id += 1

# step 0: delete old logs
log_file = CWD.joinpath(f'{INFO.program}.{PROCESS_ID}.log') if PROCESS_ID else CWD.joinpath(f'{INFO.program}.log')
for _ in glob.glob(os.path.realpath(CWD.joinpath(f'{log_file}*'))):
    try:
        os.remove(CWD.joinpath(_))
    except FileNotFoundError:
        pass
    except Exception as _:
        warnings.warn(traceback.format_exc())
        log_path = CWD.joinpath(f'{INFO.program}.log')


class _Profiler(object):
    """
    Realtime profiler with output to logger
    """

    class Task(object):
        class Status(Enum):
            Done = 'Done'
            Working = 'Working'

        def __init__(self, name, task, task_id):
            self.name = name
            self.task_id = task_id
            self.task = task

            self.trace = None
            self.start_time = time.time()
            self.end_time = None

        def __hash__(self):
            return self.task_id.__hash__()

        def __repr__(self):
            return f'<Profiler.Task>{{id: {self.task_id}, time_cost {self.time_cost:,.3f}s}}'

        def __call__(self, *args, **kwargs):
            _ = self.task(*args, **kwargs)
            self.end_time = time.time()
            return _

        @property
        def time_cost(self) -> float:
            if self.end_time is None:
                time_cost = time.time() - self.start_time
            else:
                time_cost = self.end_time - self.start_time
            return time_cost

        @property
        def status(self):
            if self.end_time:
                return self.Status.Done
            else:
                return self.Status.Working

    def __init__(self, fmt=None, logger=None, **kwargs):
        self.fmt = fmt
        self.logger = logger if logger else LOGGER.getChild('Profiler')
        self.enabled = kwargs.pop('enabled', True)
        self.log_path = kwargs.pop('log_path', CWD.joinpath(f'profile.{PROCESS_ID}.log') if PROCESS_ID else CWD.joinpath(f'profile.log'))
        self.profile_path = kwargs.pop('profile_path', CWD.joinpath(f'profile.{PROCESS_ID}.csv') if PROCESS_ID else CWD.joinpath(f'profile.csv'))
        self.report_schedule = kwargs.pop('report_schedule', 60)
        self.report_size = kwargs.pop('report_size', 10)
        self.log_level = kwargs.pop('log_level', 5)
        self.with_trace = kwargs.pop('with_trace', False)

        self.by_name: dict[str, dict[str, _Profiler.Task]] = defaultdict(dict)
        self.by_trace: dict[str, dict[str, _Profiler.Task]] = defaultdict(dict)
        self._report_status = defaultdict(dict)
        self._tasks = {}
        self._task_id = 0.

        if self.log_path and self.enabled:
            self.logger.setLevel(self.log_level)
            fh = logging.FileHandler(filename=self.log_path)
            fh.setLevel(self.log_level)
            self.logger.addHandler(fh)

        if self.report_schedule and self.enabled:
            self._report_thread = threading.Thread(target=self._report_loop, name='Profiler')
            self._report_thread.start()

        if self.log_level not in [logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG, logging.NOTSET]:
            logging.addLevelName(level=self.log_level, levelName='Profiler')

    @classmethod
    def _get_trace(cls):
        stack_list = inspect.stack()
        skip = True

        for stack in stack_list:
            file_path = pathlib.Path(stack.filename)
            if file_path == FILE_PATH and stack.function == 'profiler_decorator_wrapper':
                if skip:
                    skip = False
                elif 't' in stack.frame.f_locals:
                    trace = stack.frame.f_locals['t'].task_id
                    return trace
                else:
                    return 'root'

        return 'root'

    def _get_name(self, function):
        func_name = function.__name__

        try:
            file_name = function.__code__.co_filename
            line_num = function.__code__.co_firstlineno
        except Exception as _:
            self.logger.warning(traceback.format_exc())
            file_name = inspect.getfile(function)
            line_num = inspect.getsourcelines(function)[1]

        return f'[<{file_name}>: {line_num}; "{func_name}"]'

    def _add_task(self, name, task) -> '_Profiler.Task':
        # name = self._get_name(function=task)
        self._task_id += 1
        decorated_task = self.Task(name=name, task=task, task_id=self._task_id)

        if self.with_trace:
            trace_id = self._get_trace()
            if trace_id != 'root':
                trace_task = self._tasks[trace_id]
                decorated_task.trace = trace_task

            self.by_trace[trace_id][decorated_task.task_id] = decorated_task
            self._tasks[decorated_task.task_id] = decorated_task

        self.by_name[name][decorated_task.task_id] = decorated_task
        return decorated_task

    def __call__(self, task, **kwargs) -> '_Profiler.Task':
        self.logger.warning('use decorator mode instead!')
        name = self._get_name(function=task)
        return self._add_task(name=name, task=task)

    def _formatter(self, name: str) -> str:
        report = self._report_status[name]

        if self.fmt:
            return self.fmt(name, report)

        ttl_call = self._report_status[name]['ttl_call']
        cumulative = self._report_status[name]['cumulative']
        working_call = self._report_status[name]['working_call']
        working_cumulative = self._report_status[name]['working_cumulative']
        cps = self._report_status[name]['cps']
        done_avg = cumulative / ttl_call if ttl_call else float('nan')
        # working_avg = working_cumulative / working_num if working_num else float('nan')
        ttl_call += working_call
        new_call = self._report_status[name]['new_call']
        new_cumulative = self._report_status[name]['new_cumulative']

        if self.report_schedule:
            info = f'{name}\t{{ttl_call: {ttl_call:,}; new_call: {new_call:,}; new_cum: {new_cumulative:,.3f}s; cps: {cps:,.2f} c/s; cum: {cumulative:,.3f}s; working: {working_cumulative:,.3f}s; avg: {done_avg:,.3f}s}}'
        else:
            info = f'{name}\t{{ttl_call: {ttl_call:,}; cumulative: {cumulative:,.3f}s; working: {working_cumulative:,.3f}s; avg: {done_avg:,.3f}s}}'

        return info

    def _refresh(self, name: str | list[str] = None):
        if name is None:
            name_list = list(self.by_name)
        elif isinstance(name, str):
            name_list = [name]
        else:
            name_list = name

        for name in name_list:
            task_dict = self.by_name.get(name)
            monitor = {}
            done_num = self._report_status[name].get('ttl_call', 0)
            done_cumulative = self._report_status[name].get('cumulative', 0.)
            new_call = 0
            new_cumulative = 0.
            working_num = 0
            working_cumulative = 0.

            for task_id in list(task_dict):
                task = task_dict.pop(task_id, None)

                if task is not None:
                    if task.status == self.Task.Status.Done:
                        new_call += 1
                        done_num += 1
                        new_cumulative += task.time_cost
                        done_cumulative += task.time_cost
                    elif task.status == self.Task.Status.Working:
                        working_num += 1
                        working_cumulative += task.time_cost
                        monitor[task_id] = task

            # self.by_name[name].clear()
            self.by_name[name].update(monitor)
            self._report_status[name]['ttl_call'] = done_num
            self._report_status[name]['cumulative'] = done_cumulative
            self._report_status[name]['working_call'] = working_num
            self._report_status[name]['working_cumulative'] = working_cumulative
            self._report_status[name]['new_call'] = new_call
            self._report_status[name]['new_cumulative'] = new_cumulative
            self._report_status[name]['cps'] = new_call / self.report_schedule if self.report_schedule else float('nan')

    def _report_loop(self):
        while True:
            self._refresh()
            self._report(sort_key='new_cumulative', max_report=self.report_size)
            self._dump()
            time.sleep(self.report_schedule)

    def _report(self, sort_key: str = None, max_report: int = None):
        report_str_prefix = '--- profile ---\n'
        report_str_list = []
        if sort_key:
            sorted_name = sorted(self._report_status, key=lambda x: self._report_status[x][sort_key], reverse=True)
        else:
            sorted_name = list(self._report_status)

        entry = 0.

        for name in sorted_name:
            if self._report_status[name].get('new_call', 0):
                report_str_list.append(self._formatter(name))
                entry += 1

        if max_report is not None and len(report_str_list) >= max_report:
            report_str_suffix = f'... and {len(report_str_list) - max_report} more task ...'
            report_str_list = report_str_list[:max_report]
        else:
            report_str_suffix = '--- profile end ---'

        if report_str_list:
            report_str = report_str_prefix + '\n'.join(report_str_list) + '\n' + report_str_suffix
            self.logger.log(level=self.log_level, msg=report_str)

    def _dump(self, path: str = None, sort_key='cumulative'):
        report_df = pd.DataFrame(self._report_status).T
        try:
            report_df['avg'] = report_df['cumulative'] / report_df['ttl_call']
            report_df.sort_values(by=sort_key, inplace=True, ascending=False)
        except:
            pass

        if path is None:
            report_df.to_csv(self.profile_path)
        elif path:
            report_df.to_csv(path)

        return report_df

    @property
    def profile(self):
        """
        profiler decorator function
        """

        def decorator(function):
            if not self.enabled:
                return function

            name = self._get_name(function=function)

            if name.startswith(f'[<{FILE_PATH}>: '):
                return function
            else:
                @functools.wraps(function)
                def profiler_decorator_wrapper(*args, **kwargs):
                    t = self._add_task(name=name, task=function)
                    _ = t(*args, **kwargs)
                    if not self.report_schedule:
                        self._refresh(name=t.name)
                        self._report()
                    return _

                return profiler_decorator_wrapper

        return decorator

    @property
    def profile_all(self):
        def class_decorator(cls):
            for name, method in inspect.getmembers(cls, inspect.isfunction):
                setattr(cls, name, self.profile(method))
            return cls

        return class_decorator


LOGGER = logging.getLogger(INFO.program)
LOGGER.setLevel(LOG_LEVEL)
logging.Formatter.converter = time.gmtime
logger_ch = logging.StreamHandler(stream=sys.stdout)
logger_ch.setLevel(level=LOG_LEVEL)
logger_ch.setFormatter(fmt=ColoredFormatter())
LOGGER.addHandler(logger_ch)

logger_fh = logging.handlers.RotatingFileHandler(filename=log_file, maxBytes=1024 * 1024 * 16, backupCount=10)
logger_fh.setLevel(level=LOG_LEVEL)
logger_fh.setFormatter(fmt=ColoredFormatter())
LOGGER.addHandler(logger_fh)

# step 2: init Status and Profiler
PROFILER = _Profiler(enabled=False)

# step 3: add some annotations

# add some annotations
if GlobalStatics.TIME_ZONE is not None:
    LOGGER.warning(f'{INFO.program}.{PROCESS_ID} is using timezone={GlobalStatics.TIME_ZONE}! Some api may not support timezone awareness!')

if GlobalStatics.DEBUG_MODE:
    LOGGER.warning(f'{INFO.program}.{PROCESS_ID} is in debug mode! Limited efficiency of program execution')

__all__ = ['PROFILER', 'LOGGER', 'INFO', 'PROCESS_ID']
