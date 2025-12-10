#!/usr/bin/env python3
from __future__ import annotations

import sys
import timeit
from collections import deque
from pathlib import Path

import pytest

import support as sup

INPUT_TXT = Path(__file__).parent / "input.txt"


def compute(s: str) -> int:
    answer = 0
    for line in s.splitlines():
        target, buttons = _parse_line(line)
        answer += find_min_xor(target, buttons)
    return answer


def find_min_xor(target: int, buttons: list[int]) -> int:
    if target == 0:
        return 0
    if target in buttons:
        return 1

    visited = {0}
    queue = deque([(0, 0)])  # (current_value, steps)
    while queue:
        current, steps = queue.popleft()
        for button in buttons:
            new_value = current ^ button
            if new_value == target:
                return steps + 1
            if new_value not in visited:
                visited.add(new_value)
                queue.append((new_value, steps + 1))
    raise RuntimeError("Should not reach here")


def _parse_line(line: str) -> tuple[int, list[int]]:
    part1, part23 = line[1:].split("] (")
    part2, part3 = part23.rsplit(") {", 1)
    target = _scheme_to_int(part1)
    buttons = _buttons_to_int(part2, len(part1))
    return target, buttons


def _scheme_to_int(scheme: str) -> int:
    scheme = scheme.replace(".", "0").replace("#", "1")
    return int(scheme, 2)


def _buttons_to_int(buttons: str, target_len: int) -> list[int]:
    buttons = buttons.replace("(", "").replace(")", "")
    result = []
    for bset in buttons.split(" "):
        temp = [0] * target_len
        for b in bset.split(","):
            temp[int(b)] = 1
        result.append(int("".join(map(str, temp)), 2))
    return result


INPUT_S = """\
[.##.] (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}
[...#.] (0,2,3,4) (2,3) (0,4) (0,1,2) (1,2,3,4) {7,5,12,7,2}
[.###.#] (0,1,2,3,4) (0,3,4) (0,1,2,4,5) (1,2) {10,11,11,5,10,5}
"""
EXPECTED = 7


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

    assert result == 542


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
