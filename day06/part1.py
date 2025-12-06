#!/usr/bin/env python3
from __future__ import annotations

import sys
import timeit
from pathlib import Path

import pytest

import support as sup

INPUT_TXT = Path(__file__).parent / "input.txt"


def operation(op: str):
    start_val = 0 if op == "+" else 1

    def inner(next_val: int | None) -> int:
        nonlocal start_val
        if next_val is None:
            return start_val
        match op:
            case "+":
                start_val += next_val
            case "*":
                start_val *= next_val
            case _:
                raise ValueError(f"Unknown operation: {op}")
        return start_val

    return inner


def compute(s: str) -> int:
    data = []
    for line in s.splitlines():
        # remove repetitive spaces and split by space
        parts = line.strip().split()
        try:
            data.append([int(n) for n in parts])
        except ValueError:
            data.append(parts)

    cols = len(data[0])
    by_cols = [operation(op) for op in data[-1]]
    for row in data[:-1]:
        for col in range(cols):
            by_cols[col](row[col])

    answer = 0
    for func in by_cols:
        answer += func(None)

    return answer


INPUT_S = """\
123 328  51 64
 45 64  387 23
  6 98  215 314
*   +   *   +
"""
EXPECTED = 4277556


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

    assert result == 6378679666679


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
