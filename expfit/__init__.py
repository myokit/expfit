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

from ._vetting import (  # noqa
    vet_series,
)

from ._trans import (  # noqa
    UnitSquareTransform,
    ZoomTransform,
)

from ._est import (  # noqa
    estimate_initial_single,
)

from ._err import (  # noqa
    exp,
    rmse,
    SingleExponentialError,
    MultiExponentialError,
    ErrorWithFixedParameter,
)

from ._opt import (  # noqa
    fmin,
    LeastSquaresFit,
)

from ._ci import (  # noqa
    ci,
)

from ._fit import (  # noqa
    fit1,
    fitd2,
)

from ._tau import (  # noqa
    tau1,
    #taud2,
)

