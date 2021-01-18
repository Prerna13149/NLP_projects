from nose.tools import eq_, ok_
from distutils.version import LooseVersion

import nose
import sys
import numpy as np
import torch
import scipy

def setup_module():
    pass

def test_library_versions():

    # We use Python's distutils to verify versions. It seems sufficient for our purpose.
    # Alternatively, we could use pkg_resources.parse_version for a more robust parse.

    min_python = '3.6'
    min_numpy = '1.13'
    min_torch = '1.0'
    min_scipy = '1.0'

    ok_(LooseVersion(sys.version) > LooseVersion(min_python))
    ok_(LooseVersion(np.__version__) > LooseVersion(min_numpy))
    ok_(LooseVersion(torch.__version__) > LooseVersion(min_torch))
    ok_(LooseVersion(scipy.__version__) > LooseVersion(min_scipy))
