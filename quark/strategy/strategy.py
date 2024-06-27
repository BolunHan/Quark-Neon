import enum

from algo_engine.base import MarketData, TradeInstruction, TradeReport

from . import STRATEGY_ENGINE, LOGGER
from ..decision_core import DummyDecisionCore

LOGGER = LOGGER.getChild('Strategy')


class StatusCode(enum.Enum):
    error = -1
    idle = 0
    working = 1
    closing = 2
    closed = 3


class Strategy(object):
    def __init__(self, strategy_engine=None):
        self.engine = strategy_engine if strategy_engine is not None else STRATEGY_ENGINE

        # Using dummy core as default, no trading action will be triggered. To override this, a proper BoD function is needed.
        # This behavior is intentional, so that accidents might be avoided if strategy is not properly initialized.
        # Signals still can be collected with a dummy core, which is useful in backtest mode.
        self.decision_core = DummyDecisionCore()
        self.status = StatusCode.idle
        self.eod_status = {'last_unwind_timestamp': 0., 'retry_count': -1, 'status': 'idle', 'retry_interval': 30.}

    def register(self, **kwargs):
        self.engine.add_handler_safe(on_market_data=self._on_market_data)
        self.engine.add_handler_safe(on_order=self._on_order)
        self.engine.add_handler_safe(on_report=self._on_trade)
        self.engine.register()
        self.status = StatusCode.working
        return

    def pre_eod_unwind_all(self):
        self.position_tracker.unwind_all()
        self.status = StatusCode.closing
        self.eod_status['last_unwind_timestamp'] = self.mds.timestamp
        self.eod_status['status'] = 'working'
        self.eod_status['retry_count'] += 1
        self.eod_status['status'] = 'working'

    def pre_eod_check_unwind(self):
        if not self.status == StatusCode.closing:
            return

        exposure = self.position_tracker.exposure_volume
        working = self.position_tracker.working_volume
        timestamp = self.mds.timestamp

        # Scenario 0: no exposure
        if not exposure:
            self.status = StatusCode.closed
            self.eod_status['status'] = 'done'
            return

        # Scenario 1: canceling unwinding orders
        eod_status = self.eod_status['status']
        if eod_status == 'canceling':
            # Scenario 1.1: all canceled
            if not working['Long'] and not working['Short']:
                self.pre_eod_unwind_all()
            # Scenario 1.2: still canceling
            else:
                pass
            return

        # Scenario 2: working unwinding orders
        last_unwind_timestamp = self.eod_status['last_unwind_timestamp']
        retry_interval = self.eod_status['retry_interval']
        if last_unwind_timestamp + retry_interval < timestamp:
            self.position_tracker.cancel_all()
            self.eod_status['status'] = 'canceling'
            return

    def _on_market_data(self, market_data: MarketData, **kwargs):
        """
        implement your strategy here!
        """
        pass

    def _on_order(self, order: TradeInstruction, **kwargs):
        """
        implement your strategy here!
        """
        pass

    def _on_trade(self, report: TradeReport, **kwargs):
        """
        implement your strategy here!
        """
        pass

    def clear(self):
        self.engine.remove_handler_safe(on_market_data=self._on_market_data)
        self.engine.remove_handler_safe(on_order=self._on_order)
        self.engine.remove_handler_safe(on_report=self._on_trade)

        self.position_tracker.clear()
        self.mds.clear()

        self.status = StatusCode.idle
        self.eod_status.update({'last_unwind_timestamp': 0., 'retry_count': -1, 'status': 'idle', 'retry_interval': 30.})
        self.decision_core.clear()

        self.engine.unregister()

    @property
    def state(self):
        return self.decision_core.state

    @property
    def profile(self):
        return self.decision_core.profile

    @property
    def mds(self):
        return self.engine.mds

    @property
    def dma(self):
        return self.engine.dma

    @property
    def position_tracker(self):
        return self.engine.position_tracker

    @property
    def subscription(self):
        return self.engine.subscription


__all__ = ['StatusCode', 'Strategy']
