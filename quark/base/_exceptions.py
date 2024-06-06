__all__ = ['Exceptions']


class _Exceptions(object):
    def __init__(self):
        self.exceptions = {
            3001: self.NextTry,
            3002: self.MaxRetry,
            5001: self.TopicError,
        }

    def __getitem__(self, items):
        return self.exceptions.__getitem__(items)

    class NextTry(Exception):
        def __init__(self, error_message: str, error_code: int = 3001, *args, **kwargs):
            super().__init__(error_message, *args)
            self.error_code = error_code

            for kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

    class MaxRetry(Exception):
        def __init__(self, error_message: str, error_code: int = 3002, *args, **kwargs):
            super().__init__(error_message, *args)
            self.error_code = error_code

            for kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

    class TopicError(Exception):
        def __init__(self, error_message: str, error_code: int = 5001, *args, **kwargs):
            super().__init__(error_message, *args)
            self.error_code = error_code

            for kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])


Exceptions = _Exceptions()
