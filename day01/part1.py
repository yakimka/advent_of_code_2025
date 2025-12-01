#!/usr/bin/env python3
from __future__ import annotations

import sys
import timeit
from dataclasses import dataclass
from pathlib import Path

import pytest

import support as sup

INPUT_TXT = Path(__file__).parent / "input.txt"


@dataclass
class Dial:
    amount: int  # from 0 to 99

    def turn(self, direction: str, amount: int) -> int:
        if direction == "L":
            self.amount -= amount
        elif direction == "R":
            self.amount += amount

        while self.amount < 0:
            self.amount += 100
        self.amount = self.amount % 100
        return self.amount


def compute(s: str) -> int:
    result = 0
    dial = Dial(50)
    for line in s.splitlines():
        res = dial.turn(line[0], int(line[1:]))
        if res == 0:
            result += 1
    return result


INPUT_S = """\
L68
L30
R48
L5
R60
L55
L1
L99
R14
L82
"""
EXPECTED = 3


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

    assert result == 1172


def read_input() -> str:
    with open(INPUT_TXT) as f:
        return f.read()


if __name__ == "__main__":
    input_data = read_input()
    print("Answer is:     ", compute(input_data))

    if "-b" in sys.argv:
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
