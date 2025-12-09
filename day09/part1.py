#!/usr/bin/env python3
from __future__ import annotations

import sys
import timeit
from pathlib import Path

import pytest

import support as sup

INPUT_TXT = Path(__file__).parent / "input.txt"


Coords = tuple[int, int]


def cartesian_rectangle_area(coords1: Coords, coords2: Coords) -> int:
    """
    return number of points in the rectangle defined by coords1 and coords2
    """
    x1, y1 = coords1
    x2, y2 = coords2
    width = abs(x2 - x1) + 1
    height = abs(y2 - y1) + 1
    return width * height


def compute(s: str) -> int:
    coords: list[Coords] = []
    for line in s.splitlines():
        x, y = map(int, line.split(","))
        coords.append((x, y))

    max_area = 0
    for coord1 in coords:
        for coord2 in coords:
            area = cartesian_rectangle_area(coord1, coord2)
            if area > max_area:
                max_area = area
    return max_area


INPUT_S = """\
7,1
11,1
11,7
9,7
9,5
2,5
2,3
7,3
"""
EXPECTED = 50


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

    assert result == 4741848414


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
