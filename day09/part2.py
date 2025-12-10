#!/usr/bin/env python3
from __future__ import annotations

import sys
import timeit
from itertools import chain, combinations, pairwise
from pathlib import Path

import pytest

import support as sup

INPUT_TXT = Path(__file__).parent / "input.txt"


Coords = tuple[int, int]


class Rectangle:
    x1: int
    x2: int
    y1: int
    y2: int

    def __init__(self, coords1: Coords, coords2: Coords) -> None:
        x1, y1 = coords1[0], coords1[1]
        x2, y2 = coords2[0], coords2[1]
        self.x1, self.x2 = min(x1, x2), max(x1, x2)
        self.y1, self.y2 = min(y1, y2), max(y1, y2)

    def area(self) -> int:
        """
        return number of points in the rectangle defined by coords1 and coords2
        """
        width = abs(self.x2 - self.x1) + 1
        height = abs(self.y2 - self.y1) + 1
        return width * height

    def has_overlap(self, other: Rectangle) -> bool:
        return not (
            self.x2 <= other.x1
            or other.x2 <= self.x1
            or self.y2 <= other.y1
            or other.y2 <= self.y1
        )


def compute(s: str) -> int:
    points = []
    for line in s.splitlines():
        x, y = map(int, line.split(","))
        points.append((x, y))

    green_lines = [
        Rectangle(p1, p2)
        for p1, p2 in chain(pairwise(points), [(points[-1], points[0])])
    ]
    green_lines.sort(key=lambda r: r.area(), reverse=True)

    candidates = [Rectangle(p1, p2) for p1, p2 in combinations(points, 2)]
    candidates.sort(key=lambda r: r.area(), reverse=True)

    max_area = 0
    last_change = 0
    for rect in candidates:
        if last_change > 10:
            break

        if any(rect.has_overlap(line) for line in green_lines):
            continue
        area = rect.area()
        if area > max_area:
            max_area = area
            last_change = 0
        else:
            last_change += 1

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
EXPECTED = 24


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

    assert result == 1508918480


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
