"""Shared test setup.

Tests run offline: astropy is forbidden from touching the internet, so any test
that would need a download (for example fresh Earth orientation data) fails
instead of quietly fetching it. That is how the suite checks that the frames
module is reproducible offline.
"""

from pathlib import Path

import pytest
from astropy.utils.data import conf as astropy_data_conf

FIXTURES = Path(__file__).parent / "fixtures"
REPO = Path(__file__).resolve().parents[1]

astropy_data_conf.allow_internet = False


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES
