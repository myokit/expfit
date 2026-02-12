#!/usr/bin/env python3
#
# Test module
#
# This file is part of ExpFit.
# See https://github.com/myokit/expfit for copyright, sharing, and licensing.
#
#import os
#import tempfile
#import unittest
#import warnings

#import expfit


# The test directory
#DIR_TEST = os.path.abspath(os.path.dirname(__file__))

# The data directory
#DIR_DATA = os.path.join(DIR_TEST, 'data')

# Extra files in the data directory for load/save testing
#DIR_IO = os.path.join(DIR_DATA, 'io')

# Extra files in the data directory for format testing
#DIR_FORMATS = os.path.join(DIR_DATA, 'formats')

'''
class WarningCollector:
    """
    Wrapper around warnings.catch_warnings() that gathers all messages into a
    single string.
    """
    def __init__(self):
        self._warnings = []
        self._w = warnings.catch_warnings(record=True)

    def __enter__(self):
        self._warnings = self._w.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        self._w.__exit__(type, value, traceback)

    def count(self):
        """Returns the number of warnings caught."""
        return len(self._warnings)

    def has_warnings(self):
        """Returns ``True`` if there were any warnings."""
        return len(self._warnings) > 0

    def text(self):
        """Returns the text of all gathered warnings."""
        return ' '.join(str(w.message) for w in self._warnings)

    def warnings(self):
        """Returns all gathered warning objects."""
        return self._warnings
'''
