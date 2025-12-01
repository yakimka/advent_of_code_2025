import pytest

from support import humanized_seconds


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (12.3456789, "12.35s"),
        (0.123456789, "123ms"),
        (0.0123456789, "12ms"),
        (0.00123456789, "1ms"),
        (0.000123456789, "123μs"),
        (0.000000123456789, "123ns"),
    ],
)
def test_humanized_seconds(seconds, expected):
    assert humanized_seconds(seconds) == expected
