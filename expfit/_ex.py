#
# Exceptions specific to expfit.
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#


class NotExponentialError(RuntimeError):
    """
    Raised if no exponential can be found in the provided time series.
    """
    def __init__(self, msg=None):
        end = f': {msg}' if msg is not None else '.'
        super().__init__(f'No exponential found in time series{end}')


class NotDecayingError(RuntimeError):
    """
    Raised if a decaying exponential was expected but an expanding one was
    found.
    """
    def __init__(self, msg=None):
        end = f': {msg}' if msg is not None else '.'
        super().__init__(
            f'Exponential found in time series is not decaying{end}')


class CIUnavailableError(RuntimeError):
    """
    Raised if confidence intervals are requested but the fit result does not
    support this.
    """
    def __init__(self):
        super().__init__('CI methods unavailable for this exponential fit')

