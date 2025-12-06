#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import timeit
from pathlib import Path

import pytest

import support as sup

INPUT_TXT = Path(__file__).parent / "input.txt"


op_map = {
    "+": sum,
    "*": math.prod,
}


def compute(s: str) -> int:
    lines = s.splitlines()
    i = len(lines[0]) - 1
    last_line = len(lines) - 1
    rotated = [[]]
    delimeter = " "
    while i >= 0:
        for li, line in enumerate(lines):
            try:
                num = line[i]
            except IndexError:
                num = " "

            if li == last_line:
                if num in "+*":
                    rotated[-1].append(num)
                    rotated.append([])
                else:
                    if rotated[-1]:
                        rotated[-1].append(delimeter)
            elif num != " ":
                rotated[-1].append(num)
        i -= 1

    answer = 0
    for row in rotated:
        if not row:
            continue
        op = op_map[row[-1]]
        nums = [int(n) for n in "".join(row[:-1]).split(delimeter)]
        answer += op(nums)
    return answer


INPUT_S = """\
123 328  51 64
 45 64  387 23
  6 98  215 314
*   +   *   +
"""
EXPECTED = 3263827


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

    assert result == 11494432585168


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
