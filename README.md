# Advent of Code 2025

My solutions for https://adventofcode.com/2025/

## Setup

### Setup venv

```bash
make venv
```

Note: mark directory "support" as sources root in your IDE to get code completion for helper functions.

### Create new day from template

```bash
make new-day day=08
# or just
make new-day  # for today date
```

### Other commands

#### In root directory

```bash
# Run linters
make lint
# Run tests in all days
make test
# Generate markdown table with benchmarks
make benchmark
# Print only benchmark results
make benchmark off-formatting=1
```

#### In day directory

```bash
# Download input for a day
aoc-download-input
# Submit solution
pytest part1.py -k "not test_input" && python part1.py | aoc-submit --part=1
pytest part2.py -k "not test_input" && python part2.py | aoc-submit --part=2
```
