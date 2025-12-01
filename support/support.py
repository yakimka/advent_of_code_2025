from __future__ import annotations

import argparse
import contextlib
import heapq
import os.path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Generator,
    Hashable,
    Iterable,
    Iterator,
    NamedTuple,
    TextIO,
    TypeVar,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def _get_cookie_headers() -> dict[str, str]:
    with open(os.path.join(HERE, "../.env")) as f:
        contents = f.read().strip()
    return {"Cookie": contents, "User-Agent": "Merry Christmas!"}


def get_input(year: int, day: int) -> str:
    url = f"https://adventofcode.com/{year}/day/{day}/input"
    req = urllib.request.Request(url, headers=_get_cookie_headers())
    return urllib.request.urlopen(req).read().decode()


def get_year_day() -> tuple[int, int]:
    cwd = os.getcwd()
    day_s = os.path.basename(cwd)
    year_s = os.path.basename(os.path.dirname(cwd))

    if not day_s.startswith("day") or not year_s.startswith("advent_of_code_"):
        raise AssertionError(f"unexpected working dir: {day_s=} {year_s=}")
    year_index = len("advent_of_code_")
    day_index = len("day")
    return int(year_s[year_index:]), int(day_s[day_index:])


def download_input() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    year, day = get_year_day()

    for i in range(5):
        try:
            s = get_input(year, day)
        except urllib.error.URLError as e:
            print(f"zzz: not ready yet: {e}")
            time.sleep(1)
        else:
            break
    else:
        raise SystemExit("timed out after attempting many times")

    with open("input.txt", "w") as f:
        f.write(s)

    lines = s.splitlines()
    if len(lines) > 10:
        for line in lines[:10]:
            print(line)
        print("...")
    else:
        print(lines[0][:80])
        print("...")

    return 0


TOO_QUICK = re.compile("You gave an answer too recently.*to wait.")
WRONG = re.compile(r"That's not the right answer.*?\.")
RIGHT = "That's the right answer!"
ALREADY_DONE = re.compile(r"You don't seem to be solving.*\?")


def _post_answer(year: int, day: int, part: int, answer: int) -> str:
    params = urllib.parse.urlencode({"level": part, "answer": answer})
    req = urllib.request.Request(
        f"https://adventofcode.com/{year}/day/{day}/answer",
        method="POST",
        data=params.encode(),
        headers=_get_cookie_headers(),
    )
    resp = urllib.request.urlopen(req)

    return resp.read().decode()


def submit_solution() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, required=True)
    args = parser.parse_args()

    year, day = get_year_day()
    answer = int("".join(char for char in sys.stdin.read() if char.isdigit()))

    print(f"answer: {answer}")

    contents = _post_answer(year, day, args.part, answer)

    for error_regex in (WRONG, TOO_QUICK, ALREADY_DONE):
        error_match = error_regex.search(contents)
        if error_match:
            print(f"\033[41m{error_match[0]}\033[m")
            return 1

    if RIGHT in contents:
        print(f"\033[42m{RIGHT}\033[m")
        return 0
    else:
        # unexpected output?
        print(contents)
        return 1


def submit_25_pt2() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    year, day = get_year_day()

    assert day == 25, day
    contents = _post_answer(year, day, part=2, answer=0)

    if "Congratulations!" in contents:
        print("\033[42mCongratulations!\033[m")
        return 0
    else:
        print(contents)
        return 1


def humanized_seconds(seconds: float) -> str:
    """Convert seconds to human-readable format."""

    if seconds >= 1:
        return f"{seconds:.2f}s"
    elif seconds >= 0.001:
        return f"{seconds * 1000:.0f}ms"
    elif seconds >= 0.000_001:
        return f"{seconds * 1_000_000:.0f}μs"
    else:
        return f"{seconds * 1_000_000_000:.0f}ns"


@contextlib.contextmanager
def timed(label: str) -> Generator[None, None, None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"{label}: {humanized_seconds(end - start)}")


def print_matrix(matrix: list[list[Any]], file: TextIO | None = None) -> None:
    if not matrix or not matrix[0]:
        print("Empty matrix")
        return

    # Find the maximum length of any item in the matrix for formatting.
    max_item_len = max(len(str(item)) for row in matrix for item in row)

    # Determine the width for row and column labels.
    row_label_width = len(str(len(matrix) - 1))
    col_label_width = max(max_item_len, len(str(len(matrix[0]) - 1)))

    # Print column headers.
    col_headers = [" " * row_label_width] + [
        f"{i:<{col_label_width}}" for i in range(len(matrix[0]))
    ]
    if file is None:
        # For cases when printing in pytest or etc.
        print()
    print(" ".join(col_headers), file=file)

    # Print each row with its label.
    for i, row in enumerate(matrix):
        row_str = [f"{i:<{row_label_width}}"] + [
            f"{str(item):<{col_label_width}}" for item in row
        ]
        print(" ".join(row_str), file=file)


# ========================== helpers ==========================
def iter_lines_as_numbers(s: str) -> Generator[int, None, None]:
    for line in s.strip().splitlines():
        yield int(line)


class Range:
    __slots__ = ("start", "end")

    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end
        if self.start >= self.end:
            raise ValueError(f"{self.start=} must be < {self.end=}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.start}, {self.end})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Range):
            return NotImplemented
        return self.start == other.start and self.end == other.end

    def __contains__(self, n: int) -> bool:
        return self.start <= n < self.end

    def __len__(self) -> int:
        return self.end - self.start

    def has_intersection(self, other: Range) -> bool:
        return self.start < other.end and other.start < self.end

    def intersection(self, other: Range) -> Range | None:
        if not self.has_intersection(other):
            return None
        return Range(max(self.start, other.start), min(self.end, other.end))

    def remainder(self, other: Range) -> list[Range]:
        intersection = self.intersection(other)
        if intersection is None:
            return []

        result = []
        if self.start < intersection.start:
            result.append(Range(self.start, intersection.start))
        if intersection.end < self.end:
            result.append(Range(intersection.end, self.end))
        return result


class Vector2D(NamedTuple):
    x: int
    y: int

    def __add__(self, other: Vector2D | tuple[int, int]) -> Vector2D:
        x, y = other
        return Vector2D(self.x + x, self.y + y)

    def __mul__(self, other: int) -> Vector2D:
        return Vector2D(self.x * other, self.y * other)


class Direction(Vector2D, Enum):
    UP = Vector2D(-1, 0)
    DOWN = Vector2D(1, 0)
    LEFT = Vector2D(0, -1)
    RIGHT = Vector2D(0, 1)
    UPLEFT = Vector2D(-1, -1)
    UPRIGHT = Vector2D(-1, 1)
    DOWNLEFT = Vector2D(1, -1)
    DOWNRIGHT = Vector2D(1, 1)


Coords = tuple[int, int]
inf_coords = (float("inf"), float("inf"))
T = TypeVar("T")


@dataclass
class Matrix:
    data: list[list[T]]

    @property
    def m_len(self) -> int:
        return len(self.data)

    @property
    def n_len(self) -> int:
        return len(self.data[0])

    @property
    def bounds(self) -> tuple[int, int]:
        return self.m_len - 1, self.n_len - 1

    @classmethod
    def create_from_input(
        cls, s: str, *, split_by: str = "", cast_func: Callable[[str], T] = str
    ) -> Matrix:
        matrix = []
        for line in s.strip().splitlines():
            if split_by:
                line = line.split(split_by)
            if cast_func is not str:
                matrix.append([cast_func(item) for item in line])
            else:
                matrix.append(list(line))

        return cls(matrix)

    def __iter__(self) -> Iterator[list[T]]:
        return iter(self.data)

    def __getitem__(self, m: int) -> list[T]:
        return self.data[m]

    def copy(self) -> Matrix:
        return Matrix([row.copy() for row in self.data])

    def neighbors_cross(self, m: int, n: int) -> Generator[Coords, None, None]:
        neighbors = (
            (m, n - 1),
            (m - 1, n),
            (m + 1, n),
            (m, n + 1),
        )
        yield from filter_neighbors(neighbors, max_bounds=self.bounds)

    def neighbors_diag(self, m: int, n: int) -> Generator[Coords, None, None]:
        neighbors = (
            (m - 1, n - 1),
            (m + 1, n - 1),
            (m - 1, n + 1),
            (m + 1, n + 1),
        )
        yield from filter_neighbors(neighbors, max_bounds=self.bounds)

    def neighbors_cross_diag(self, m: int, n: int) -> Generator[Coords, None, None]:
        yield from self.neighbors_cross(m, n)
        yield from self.neighbors_diag(m, n)

    def neighbors_cross_diag_all(
        self, m: int, n: int, *, default=None
    ) -> Generator[Coords | None, None, None]:
        """
        Return all neighbors, including out of bounds.
        Clockwise order (from up-left corner).
        """
        neighbors = (
            (m - 1, n - 1),
            (m - 1, n),
            (m - 1, n + 1),
            (m, n + 1),
            (m + 1, n + 1),
            (m + 1, n),
            (m + 1, n - 1),
            (m, n - 1),
        )
        max_bounds = self.bounds
        yield from (
            (m, n) if 0 <= m <= max_bounds[0] and 0 <= n <= max_bounds[1] else default
            for m, n in neighbors
        )

    def next_coords(
        self, m: int, n: int, direction: Vector2D, size: int = 1
    ) -> Coords | None:
        next_m, next_n = m + direction.x * size, n + direction.y * size
        if (
            0 > next_m
            or 0 > next_n
            or next_m > self.bounds[0]
            or next_n > self.bounds[1]
        ):
            return None

        return next_m, next_n

    def get_values(self, m: int, n: int, direction: Vector2D, size: int = 2) -> list[T]:
        results = []
        for i in range(size):
            next_m, next_n = m + direction.x * i, n + direction.y * i
            if next_m < 0 or next_n < 0:
                return results

            try:
                results.append(self.data[next_m][next_n])
            except IndexError:
                return results
        return results


def filter_neighbors(
    neighbors: Iterable[Coords], *, max_bounds: Coords = inf_coords
) -> Generator[Coords, None, None]:
    yield from (
        (m, n)
        for m, n in neighbors
        if 0 <= m <= max_bounds[0] and 0 <= n <= max_bounds[1]
    )


def cartesian_shortest_path(coords1: Coords, coords2: Coords) -> int:
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


HT = TypeVar("HT", bound=Hashable)


def bfs(graph: dict[HT, dict[HT, int]], source: HT) -> dict[HT, HT]:
    queue = deque([source])
    prev = {source: None}

    while queue:
        u = queue.popleft()
        for vertex, val in graph[u].items():
            if vertex not in prev:
                queue.append(vertex)
                prev[vertex] = u
    return prev


def dijkstra(
    graph: dict[HT, dict[HT, int]], source: HT
) -> tuple[dict[HT, int], dict[HT, HT]]:
    dist = {source: 0}
    prev = {source: None}
    pq = [(0, source)]
    while pq:
        _, u = heapq.heappop(pq)
        for vertex, val in graph[u].items():
            if vertex not in dist or dist[u] + val < dist[vertex]:
                dist[vertex] = dist[u] + val
                prev[vertex] = u
                heapq.heappush(pq, (dist[vertex], vertex))
    return dist, prev


def a_star(
    graph: dict[HT, dict[HT, int]],
    source: HT,
    target: HT,
    heuristic: Callable[[HT, HT], int],
) -> tuple[dict[HT, int], dict[HT, HT]]:
    """
    A* algorithm implementation.

    heuristic function must be admissible (never overestimate the distance to the goal).
    https://en.wikipedia.org/wiki/A*_search_algorithm#Admissibility
    for example, Manhattan distance is admissible.
    >>> def heuristic(candidate, target):
    ...     (x1, y1) = candidate
    ...     (x2, y2) = target
    ...     return abs(x1 - x2) + abs(y1 - y2)

    :param graph: graph in format {vertex: {neighbor: cost}}
    :param source: source vertex
    :param target: target vertex
    :param heuristic: heuristic function
    :return: tuple of distance and previous vertex
    """
    dist = {source: 0}
    prev = {source: None}
    pq = [(0, source)]

    while pq:
        _, current = heapq.heappop(pq)

        if current == target:
            break

        for vertex, val in graph[current].items():
            new_cost = dist[current] + val
            if vertex not in dist or new_cost < dist[vertex]:
                dist[vertex] = new_cost
                priority = new_cost + heuristic(vertex, target)
                heapq.heappush(pq, (priority, vertex))
                prev[vertex] = current

    return dist, prev


def topological_sort(graph: dict[HT, list[HT]]) -> list[HT]:
    # https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
    result = []
    temp_marks = set()
    perm_marks = set()

    def visit(node: HT) -> None:
        if node in perm_marks:
            return
        if node in temp_marks:
            raise ValueError("graph has at least one cycle")

        temp_marks.add(node)
        for neighbor in graph.get(node, []):
            visit(neighbor)
        temp_marks.remove(node)
        perm_marks.add(node)
        result.append(node)

    for node in graph:
        visit(node)

    return result


class GraphCycleFinder:
    def __init__(self, graph: dict[HT, list[HT]] | None = None) -> None:
        self.graph = graph or {}
        self.cliques = []

    def add_edge(self, u: HT, v: HT) -> None:
        self.graph.setdefault(u, []).append(v)
        self.graph.setdefault(v, []).append(u)

    def bron_kerbosch(self, r: set[HT], p: set[HT], x: set[HT]):
        """
        Recursive Bron–Kerbosch algorithm.

        https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm
        """
        if not p and not x:
            self.cliques.append(r)
            return

        for v in list(p):
            self.bron_kerbosch(r | {v}, p & set(self.graph[v]), x & set(self.graph[v]))
            p.remove(v)
            x.add(v)

    def find_cliques(self) -> list[HT]:
        """Find all maximal cliques in the graph."""
        self.cliques = []
        nodes = set(self.graph.keys())
        self.bron_kerbosch(set(), nodes, set())
        return self.cliques

    def find_cycles(self) -> list[HT]:
        """Filter cliques to find those that form cycles."""
        self.find_cliques()
        cycles = []

        for clique in self.cliques:
            if len(clique) >= 3:
                # Verify that the clique forms a cycle
                is_cycle = True
                for node in clique:
                    neighbors = set(self.graph[node])
                    if any(n not in neighbors for n in clique if n != node):
                        is_cycle = False
                        break
                if is_cycle:
                    cycles.append(clique)

        return cycles
