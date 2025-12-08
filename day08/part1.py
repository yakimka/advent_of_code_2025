#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import timeit
from itertools import combinations
from pathlib import Path

import pytest

import support as sup

INPUT_TXT = Path(__file__).parent / "input.txt"


def compute(s: str, count: int = 1000) -> int:
    points = [tuple(map(int, item.split(","))) for item in s.splitlines()]
    connections = sorted(combinations(points, 2), key=lambda p: math.dist(*p))

    ds = sup.DisjointSet(points)
    for conn in connections:
        if conn == connections[count]:
            return math.prod(sorted(map(len, ds.subsets()))[-3:])
        ds.merge(*conn)

    raise ValueError("Should not reach here")


INPUT_S = """\
162,817,812
57,618,57
906,360,560
592,479,940
352,342,300
466,668,158
542,29,236
431,825,988
739,650,466
52,470,668
216,146,977
819,987,18
117,168,530
805,96,715
346,949,466
970,615,88
941,993,340
862,61,35
984,92,344
425,690,689
"""
EXPECTED = 40


@pytest.mark.parametrize(
    "input_s,expected",
    [
        (INPUT_S, EXPECTED),
    ],
)
def test_debug(input_s: str, expected: int) -> None:
    assert compute(input_s, count=10) == expected


def test_input() -> None:
    result = compute(read_input())

    assert result == 62186


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
