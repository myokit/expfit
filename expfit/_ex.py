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


class NotOpposingError(RuntimeError):
    """
    Raised if two opposing (opposite sign) exponentials are requested, but not
    found in the signal.
    """
    def __init__(self, msg=None):
        super().__init__('No opposing exponentials found in the signal.')


class CIUnavailableError(RuntimeError):
    """
    Raised if confidence intervals are requested but the fit result does not
    support this.
    """
    def __init__(self):
        super().__init__('CI methods unavailable for this exponential fit')


class CILimitNotFound(RuntimeError):
    """
    Raised if the ``ci_profile`` method cannot find an upper or lower bound
    near the expected value.
    """
    def __init__(self, direction):
        direction = 'upper' if direction > 0 else 'lower'
        super().__init__(
            f'Unable to find {direction} limit during method `ci_profile`:'
            ' expansion reached maximum iterations, or went beyond 10 times'
            ' the FIM estimate.')
