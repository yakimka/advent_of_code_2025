#!/usr/bin/env python3
from __future__ import annotations

import sys
import timeit
from collections import defaultdict
from pathlib import Path

import pytest

import support as sup

INPUT_TXT = Path(__file__).parent / "input.txt"


def compute(s: str) -> int:
    matrix = sup.Matrix.create_from_input(s)
    start_pos = 0, next(i for i, n in enumerate(matrix[0]) if n == "S")
    cache = defaultdict(int)
    cache[start_pos] += 1
    answer = 1
    for _ in range(matrix.rows - 1):
        temp = defaultdict(int)
        for (m, n), count in cache.items():
            if m == matrix.rows - 1:
                break
            if matrix[m + 1][n] == "^":
                answer += count
                temp[(m + 1, n - 1)] += count
                temp[(m + 1, n + 1)] += count
            else:
                temp[(m + 1, n)] += count
        cache = temp
    return answer


INPUT_S = """\
.......S.......
...............
.......^.......
...............
......^.^......
...............
.....^.^.^.....
...............
....^.^...^....
...............
...^.^...^.^...
...............
..^...^.....^..
...............
.^.^.^.^.^...^.
...............
"""
EXPECTED = 40


@pytest.mark.parametrize(
    "input_s,expected",
    [
        (INPUT_S, EXPECTED),
    ],
)
def test_debug(input_s: str, expected: int) -> None:
    assert compute(input_s) == expected


def test_input() -> None:
    result = compute(read_input())

    assert result == 32982105837605


def read_input() -> str:
    with open(INPUT_TXT) as f:
        return f.read()


if __name__ == "__main__":
    input_data = read_input()
    if "-bh" not in sys.argv:
        print("Answer is:     ", compute(input_data))

    if "-b" in sys.argv or "-bh" in sys.argv:
        number_of_runs = 100
        bench_time = timeit.timeit(
            "compute(data)",
            setup="from __main__ import compute",
            globals={"data": input_data},
            number=number_of_runs,
        )
        print(f"{number_of_runs} runs took: {bench_time}s")
        one_run = sup.humanized_seconds(bench_time / number_of_runs)
        print(f"Average time:   {one_run}")
