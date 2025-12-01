from functools import partial

import pytest

import support as sup


@pytest.fixture()
def matrix():
    return sup.Matrix(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )


@pytest.fixture()
def graph(matrix):
    return {
        (m, n): {
            (n_m, n_n): matrix[n_m][n_n] for n_m, n_n in matrix.neighbors_cross(m, n)
        }
        for m in range(matrix.m_len)
        for n in range(matrix.n_len)
    }


@pytest.mark.parametrize(
    "function",
    [
        sup.dijkstra,
        partial(sup.a_star, target=(2, 2), heuristic=lambda a, b: 0),
    ],
)
def test_pathfinding_functions_with_weights(graph, function) -> None:
    source = (0, 0)

    result_dist, result_prev = function(graph, source)

    assert result_dist == {
        (0, 0): 0,
        (0, 1): 2,
        (0, 2): 5,
        (1, 0): 4,
        (1, 1): 7,
        (1, 2): 11,
        (2, 0): 11,
        (2, 1): 15,
        (2, 2): 20,
    }
    assert result_prev == {
        (0, 0): None,
        (0, 1): (0, 0),
        (0, 2): (0, 1),
        (1, 0): (0, 0),
        (1, 1): (0, 1),
        (1, 2): (0, 2),
        (2, 0): (1, 0),
        (2, 1): (1, 1),
        (2, 2): (1, 2),
    }


@pytest.mark.parametrize(
    "function",
    [
        sup.bfs,
    ],
)
def test_pathfinding_functions_wo_weights(graph, function) -> None:
    source = (0, 0)

    result_prev = function(graph, source)

    assert result_prev == {
        (0, 0): None,
        (0, 1): (0, 0),
        (0, 2): (0, 1),
        (1, 0): (0, 0),
        (1, 1): (1, 0),
        (1, 2): (1, 1),
        (2, 0): (1, 0),
        (2, 1): (2, 0),
        (2, 2): (2, 1),
    }
