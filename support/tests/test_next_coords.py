import pytest

from support import Direction, Matrix


@pytest.fixture()
def matrix():
    return Matrix(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
        ]
    )


@pytest.mark.parametrize(
    "coords,direction,expected",
    [
        ((0, 0), Direction.UP, None),
        ((0, 0), Direction.DOWN, (1, 0)),
        ((0, 0), Direction.LEFT, None),
        ((0, 0), Direction.RIGHT, (0, 1)),
        ((0, 1), Direction.UP, None),
        ((0, 1), Direction.DOWN, (1, 1)),
        ((0, 1), Direction.LEFT, (0, 0)),
        ((0, 1), Direction.RIGHT, (0, 2)),
        ((0, 4), Direction.UP, None),
        ((0, 4), Direction.DOWN, (1, 4)),
        ((0, 4), Direction.LEFT, (0, 3)),
        ((0, 4), Direction.RIGHT, None),
        ((1, 0), Direction.UP, (0, 0)),
        ((1, 0), Direction.DOWN, (2, 0)),
        ((1, 0), Direction.LEFT, None),
        ((1, 0), Direction.RIGHT, (1, 1)),
        ((1, 1), Direction.UP, (0, 1)),
        ((1, 1), Direction.DOWN, (2, 1)),
        ((1, 1), Direction.LEFT, (1, 0)),
        ((1, 1), Direction.RIGHT, (1, 2)),
        ((1, 4), Direction.UP, (0, 4)),
        ((1, 4), Direction.DOWN, (2, 4)),
        ((1, 4), Direction.LEFT, (1, 3)),
        ((1, 4), Direction.RIGHT, None),
        ((3, 0), Direction.UP, (2, 0)),
        ((3, 0), Direction.DOWN, None),
        ((3, 0), Direction.LEFT, None),
        ((3, 0), Direction.RIGHT, (3, 1)),
        ((3, 1), Direction.UP, (2, 1)),
        ((3, 1), Direction.DOWN, None),
        ((3, 1), Direction.LEFT, (3, 0)),
        ((3, 1), Direction.RIGHT, (3, 2)),
        ((3, 4), Direction.UP, (2, 4)),
        ((3, 4), Direction.DOWN, None),
    ],
)
def test_next_coords(matrix, coords, direction, expected):
    m, n = coords

    result = matrix.next_coords(m, n, direction)

    assert result == expected
