#!/usr/bin/env python3
from __future__ import annotations

import sys
import timeit
from pathlib import Path

import pytest

import support as sup

INPUT_TXT = Path(__file__).parent / "input.txt"


def compute(s: str) -> int:
    result = 0
    length = len(s.splitlines()[0])
    for line in s.splitlines():
        max_to_end = {}
        first_index, first_char = 0, ""
        for i in range(length):
            if i == length - 1:
                first_index, first_char = max_to_end[0]
            current = line[i]
            for prev_bucket in range(i + 1):
                prev_max = max_to_end.get(prev_bucket, (None, "-1"))[1]
                if current > prev_max:
                    max_to_end[prev_bucket] = (i, current)
        second_char = max_to_end[first_index + 1][1]
        joltage = int(f"{first_char}{second_char}")
        result += joltage
    return result


INPUT_S = """\
987654321111111
811111111111119
234234234234278
818181911112111
"""
EXPECTED = 357


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

    assert result == 16946


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
