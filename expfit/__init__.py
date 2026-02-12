#
# Expfit's main module
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
"""
ExpFit

This module .....
"""

#
# GUI and graphical modules should not be auto-included because they define a
# matplotlib backend to use. If the user requires a different backend, this
# will generate an error.
#

#
# Check python version
#
# Hexversion guide:
#  0x   Hex
#  02   PY_MAJOR_VERSION
#  07   PY_MINOR_VERSION
#  0F   PY_MICRO_VERSION, in hex, so 0F is 15, 10 is 16, etc.
#  F    PY_RELEASE_LEVEL, A for alpha, B for beta, C for candidate, F for final
#  0    PY_RELEASE_SERIAL, increments with every release
#
import sys  # noqa
if sys.hexversion < 0x03000000:  # pragma: no cover
    raise Exception('This version of Myokit does not support Python 2.')
if sys.hexversion < 0x03080000:  # pragma: no cover
    import warnings  # noqa
    warnings.warn(
        'ExpFit is not tested on Python versions before 3.8. Detected'
        f' version {sys.version}.')
    del warnings
del sys


#
# Version information
#
from ._expfit_version import (  # noqa
    #__release__,
    __version__,
    #__version_tuple__,
)


# Warn about development version
#if not __release__:     # pragma: no cover
#    import warnings  # noqa
#    warnings.warn(f'Using development version of Myokit ({__version__}).')
#    del warnings



#
# Paths
#

# Expfit root
#import os, inspect  # noqa
#try:
#    frame = inspect.currentframe()
#    DIR_MYOKIT = os.path.abspath(os.path.dirname(inspect.getfile(frame)))
#finally:
#    # Always manually delete frame
#    # https://docs.python.org/2/library/inspect.html#the-interpreter-stack
#    del frame

# Binary data files
#DIR_DATA = os.path.join(DIR_MYOKIT, '_bin')


# Location of myokit user config
#DIR_USER = os.path.join(os.path.expanduser('~'), '.config', 'myokit')

## Ensure the user config directory exists and is writable
#if os.path.exists(DIR_USER):    # pragma: no cover
#    if not os.path.isdir(DIR_USER):
#        raise Exception(
#            f'File or link found in place of user directory: {DIR_USER}')
#else:                           # pragma: no cover
#    os.makedirs(DIR_USER)


# Don't expose standard libraries as part of Myokit
#del os, inspect


#
# Imports
#

# Exceptions
#from ._err import (  # noqa



#
# Load settings
#
#from . import _config   # noqa
#del _config

