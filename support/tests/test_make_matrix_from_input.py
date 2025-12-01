import pytest

from support import Matrix


@pytest.mark.parametrize(
    "input_s",
    [
        ".#.\n###",
        "\n.#.\n###",
        ".#.\n###\n",
        "\n.#.\n###\n",
        "\n\n.#.\n###\n\n",
    ],
)
def test_make_string_matrix_from_input(input_s) -> None:
    result = Matrix.create_from_input(input_s)

    assert result.data == [[".", "#", "."], ["#", "#", "#"]]


@pytest.mark.parametrize(
    "input_s,expected",
    [
        ("1 2\n22 33\n4 78", [[1, 2], [22, 33], [4, 78]]),
    ],
)
def test_make_int_matrix_from_input(input_s, expected) -> None:
    result = Matrix.create_from_input(input_s, split_by=" ", cast_func=int)

    assert result.data == expected
