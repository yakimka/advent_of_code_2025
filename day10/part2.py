#!/usr/bin/env python3
from __future__ import annotations

import sys
import timeit
from pathlib import Path

import pytest
from z3 import IntVector, Optimize, Sum, sat

import support as sup

INPUT_TXT = Path(__file__).parent / "input.txt"


def compute(s: str) -> int:
    answer = 0
    for line in s.splitlines():
        buttons, joltage = _parse_line(line)
        answer += z3_solver(buttons, joltage)
    return answer


def z3_solver(buttons: list[list[int]], joltage: list[int]):
    unknown = IntVector("x", len(buttons))

    opt = Optimize()
    opt.add([x >= 0 for x in unknown])

    for i, target in enumerate(joltage):
        terms = []
        for j in range(len(buttons)):
            term = unknown[j] * buttons[j][i]
            terms.append(term)
        opt.add(Sum(terms) == target)

    opt.minimize(Sum(unknown))

    if opt.check() == sat:
        model = opt.model()
        return sum(model[x].as_long() for x in unknown)
    raise RuntimeError("No solution found")


def _parse_line(line: str) -> tuple[list[list[int]], list[int]]:
    part1, part23 = line.split("] (")
    part2, part3 = part23[:-1].rsplit(") {", 1)
    buttons = _parse_buttons(part2, len(part1))
    joltage = _parse_joltage(part3)
    return buttons, joltage


def _parse_buttons(buttons: str, target_len: int) -> list[list[int]]:
    buttons = buttons.replace("(", "").replace(")", "")
    result = []
    for bset in buttons.split(" "):
        temp = [0] * target_len
        for b in bset.split(","):
            temp[int(b)] = 1
        result.append(temp)
    return result


def _parse_joltage(joltage: str) -> list[int]:
    return [int(x) for x in joltage.split(",")]


INPUT_S = """\
[.##.] (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}
[...#.] (0,2,3,4) (2,3) (0,4) (0,1,2) (1,2,3,4) {7,5,12,7,2}
[.###.#] (0,1,2,3,4) (0,3,4) (0,1,2,4,5) (1,2) {10,11,11,5,10,5}
"""
EXPECTED = 33


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

    assert result == 20871


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
