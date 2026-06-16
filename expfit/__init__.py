#
# Expfit's main module
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
"""
ExpFit

This package provides functions to fit a handful of exponentials to noisy time
series data.
"""

#
# Version information
#
from ._expfit_version import (  # noqa
    __version__,
    __version_tuple__,
)

#
# Imports
#

from ._ex import (  # noqa
    CILimitNotFound,
    CIUnavailableError,
    NotDecayingError,
    NotExponentialError,
    NotOpposingError,
)

from ._vetting import (  # noqa
    vet_series,
)

from ._trans import (  # noqa
    UnitSquareTransform,
    ZoomTransform,
)

from ._est import (  # noqa
    estimate_initial_single,
    estimate_initial_opposing,
    #estimate_noise_level,
)

from ._err import (  # noqa
    ConstraintWithFixedParameter,
    ErrorWithFixedParameter,
    MultiExponentialConstraint,
    MultiExponentialError,
    SingleExponentialError,
    TauFormError,
    exp,
    expc,
    rmse,
    rmsec,
)

from ._opt import (  # noqa
    LeastSquaresFit,
    lm,
    LMResult,
)

from ._ci import (  # noqa
    CLevel,
    ExponentialFit,
)

from ._fit import (  # noqa
    fit1,
    fitd2,
    fitd11,
)

# Numpy compatibility
import numpy as np
try:
    _trapezoid = np.trapezoid
except AttributeError:  # pragma: no cover
    _trapezoid = np.trapz
del np

