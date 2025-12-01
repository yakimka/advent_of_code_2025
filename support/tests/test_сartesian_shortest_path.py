import pytest

from support import cartesian_shortest_path


@pytest.mark.parametrize(
    "coords1,coords2,expected",
    [
        ((0, 0), (0, 0), 0),
        ((-1, 0), (0, 0), 1),
        ((0, 0), (0, -1), 1),
        ((-1, -1), (1, 1), 4),
    ],
)
def test_shortest_path(
    coords1: tuple[int, int], coords2: tuple[int, int], expected: int
) -> None:
    assert cartesian_shortest_path(coords1, coords2) == expected
