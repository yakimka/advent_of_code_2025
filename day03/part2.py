#!/usr/bin/env python3
from __future__ import annotations

import sys
import timeit
from pathlib import Path

import pytest

import support as sup

INPUT_TXT = Path(__file__).parent / "input.txt"


def compute(s: str, bcount: int = 12) -> int:
    result = 0
    for line in s.splitlines():
        start = 0
        length = len(line)
        remaining = bcount
        chosen = []

        while remaining:
            search_end = length - remaining + 1
            max_char, max_index = "-1", -1
            for i in range(start, search_end):
                char = line[i]
                if char > max_char:
                    max_char = char
                    max_index = i

            chosen.append(max_char)
            start = max_index + 1
            remaining -= 1

        joltage = int("".join(chosen))
        result += joltage
    return result


INPUT_S = """\
987654321111111
811111111111119
234234234234278
818181911112111
"""
EXPECTED = 3121910778619


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

    assert result == 168627047606506


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
