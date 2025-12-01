import pytest

from support import Matrix


@pytest.fixture()
def make_matrix():
    def maker(bounds):
        max_m, max_n = bounds
        return Matrix([[0 for _ in range(max_n + 1)] for _ in range(max_m + 1)])

    return maker


@pytest.mark.parametrize(
    "coords,max_bounds,expected",
    [
        ((0, 0), (10, 10), [(1, 0), (0, 1)]),
        ((0, 0), (0, 0), []),
        ((1, 1), (10, 10), [(1, 0), (0, 1), (2, 1), (1, 2)]),
        ((1, 1), (1, 1), [(1, 0), (0, 1)]),
    ],
)
def test_neighbors_cross(make_matrix, coords, max_bounds, expected):
    matrix = make_matrix(max_bounds)
    x, y = coords

    result = matrix.neighbors_cross(x, y)

    assert list(result) == expected


@pytest.mark.parametrize(
    "coords,max_bounds,expected",
    [
        ((0, 0), (10, 10), [(1, 1)]),
        ((0, 0), (0, 0), []),
        ((1, 1), (10, 10), [(0, 0), (2, 0), (0, 2), (2, 2)]),
        ((1, 1), (1, 1), [(0, 0)]),
    ],
)
def test_neighbors_diag(make_matrix, coords, max_bounds, expected):
    matrix = make_matrix(max_bounds)
    x, y = coords

    result = matrix.neighbors_diag(x, y)

    assert list(result) == expected


@pytest.mark.parametrize(
    "coords,max_bounds,expected",
    [
        (
            (0, 0),
            (10, 10),
            [(1, 0), (0, 1), (1, 1)],
        ),
        ((0, 0), (0, 0), []),
        (
            (1, 1),
            (10, 10),
            [(1, 0), (0, 1), (2, 1), (1, 2), (0, 0), (2, 0), (0, 2), (2, 2)],
        ),
        (
            (1, 1),
            (1, 1),
            [(1, 0), (0, 1), (0, 0)],
        ),
    ],
)
def test_neighbors_cross_diag(make_matrix, coords, max_bounds, expected):
    matrix = make_matrix(max_bounds)
    x, y = coords

    result = matrix.neighbors_cross_diag(x, y)

    assert list(result) == expected
