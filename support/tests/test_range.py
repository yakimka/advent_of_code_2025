import pytest

from support import Range


@pytest.mark.parametrize(
    "candidate,expected",
    [
        (12, True),
        (33, True),
        (54, True),
        (55, False),
        (100, False),
        (0, False),
        (-1, False),
    ],
)
def test_contains(candidate, expected):
    sample_range = Range(12, 55)

    result = candidate in sample_range

    assert result == expected


@pytest.mark.parametrize(
    "candidate,expected",
    [
        (Range(12, 12 + 5), 5),
        (Range(-42, -42 + 12), 12),
        (Range(0, 7), 7),
    ],
)
def test_len(candidate, expected):
    result = len(candidate)

    assert result == expected


@pytest.mark.parametrize(
    "candidate,expected",
    [
        (Range(-14, -11), True),
        (Range(-12, 55), True),
        (Range(0, 12), True),
        (Range(-14, -12), False),
        (Range(-14, -13), False),
        (Range(55, 100), False),
        (Range(100, 112), False),
    ],
)
def test_has_intersection(candidate, expected):
    sample_range = Range(-12, 55)

    result = sample_range.has_intersection(candidate)

    assert result == expected


@pytest.mark.parametrize(
    "candidate,expected",
    [
        (Range(-14, -11), (-12, -11)),
        (Range(-12, 55), (-12, 55)),
        (Range(0, 12), (0, 12)),
    ],
)
def test_intersection__has_intersection(candidate, expected):
    sample_range = Range(-12, 55)

    result = sample_range.intersection(candidate)

    assert (result.start, result.end) == expected


@pytest.mark.parametrize(
    "candidate",
    [
        Range(-14, -12),
        Range(-14, -13),
        Range(55, 100),
        Range(100, 112),
    ],
)
def test_intersection__has_not_intersection(candidate):
    sample_range = Range(-12, -11)

    result = sample_range.intersection(candidate)

    assert result is None


@pytest.mark.parametrize(
    "candidate,expected",
    [
        (Range(-14, -13), []),
        (Range(-12, 55), []),
        (Range(-12, 50), [Range(50, 55)]),
        (Range(-1, 44), [Range(-12, -1), Range(44, 55)]),
        (Range(0, 55), [Range(-12, 0)]),
    ],
)
def test_remainder(candidate, expected):
    sample_range = Range(-12, 55)

    result = sample_range.remainder(candidate)

    assert result == expected
