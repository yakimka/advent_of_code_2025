import pytest

from support import topological_sort


@pytest.mark.parametrize(
    "graph,expected",
    [
        (
            {
                "47": ["53", "61"],
                "61": ["53"],
                "75": ["53", "47", "61"],
                "97": ["61", "47", "53", "75"],
            },
            ["53", "61", "47", "75", "97"],
        ),
    ],
)
def test_topological_sort(graph, expected):
    assert topological_sort(graph) == expected
