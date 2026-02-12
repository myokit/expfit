#
# Myokit's version info
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#

# True if this is a release, False for a development version
__release__ = False

# Version as a tuple (major, minor, revision)
#  - Changes to major are rare
#  - Changes to minor indicate new features, possible slight backwards
#    incompatibility
#  - Changes to revision indicate bugfixes, tiny new features
#  - There is no significance to odd/even numbers
__version_tuple__ = 0, 1, 0

# String version of the version number
__version__ = '.'.join([str(x) for x in __version_tuple__])
if not __release__:  # pragma: no cover
    __version_tuple__ += ('dev', )
    __version__ += '.dev'

