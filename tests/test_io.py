import pytest
from pathlib import Path

from optifik import io


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent.parent / 'data'


def test_load_file(test_data_dir):
    path = test_data_dir / 'spectraLorene' / 'sample1' / '000266.xy'
    data = io.load_spectrum(path)
    assert(len(data) == 2)
    assert(data[0].shape == data[1].shape)
