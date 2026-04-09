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
    vet_series
)

from ._linear import (  # noqa
    find_linear_segment,
    least_squares,
)

from ._single import (  # noqa
    estimate_initial_single,
    fit_single,
    rmse_single,
)

from ._double import (  # noqa
    fit_double,
    rmse_double,
)
