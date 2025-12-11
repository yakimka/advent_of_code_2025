#!/usr/bin/env python3
from __future__ import annotations

import sys
import timeit
from functools import cache
from pathlib import Path

import pytest

import support as sup

INPUT_TXT = Path(__file__).parent / "input.txt"


def compute(s: str) -> int:
    links = {"out": []}
    for line in s.splitlines():
        parent, children_s = line.split(": ")
        children = children_s.split()
        links[parent] = children

    @cache
    def count(node: str, target: str) -> int:
        if node == target:
            return 1
        return sum(count(link, target) for link in links[node])

    part1 = count("svr", "fft")
    part2 = count("fft", "dac")
    part3 = count("dac", "out")
    return part1 * part2 * part3


INPUT_S = """\
svr: aaa bbb
aaa: fft
fft: ccc
bbb: tty
tty: ccc
ccc: ddd eee
ddd: hub
hub: fff
eee: dac
dac: fff
fff: ggg hhh
ggg: out
hhh: out
"""
EXPECTED = 2


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

    assert result == 450854305019580


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
