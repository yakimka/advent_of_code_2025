import pytest

from support import Range, merge_ranges


# Happy path tests
def test_merges_overlapping_ranges():
    # Arrange
    ranges = [Range(1, 5), Range(3, 7)]

    # Act
    result = merge_ranges(ranges)

    # Assert
    assert len(result) == 1
    assert result[0] == Range(1, 7)


def test_merges_adjacent_ranges():
    # Arrange
    ranges = [Range(1, 5), Range(5, 10)]

    # Act
    result = merge_ranges(ranges)

    # Assert
    assert len(result) == 1
    assert result[0] == Range(1, 10)


def test_merges_multiple_overlapping_ranges():
    # Arrange
    ranges = [Range(1, 5), Range(3, 7), Range(5, 10)]

    # Act
    result = merge_ranges(ranges)

    # Assert
    assert len(result) == 1
    assert result[0] == Range(1, 10)


def test_keeps_separate_non_overlapping_ranges():
    # Arrange
    ranges = [Range(1, 5), Range(10, 15)]

    # Act
    result = merge_ranges(ranges)

    # Assert
    assert len(result) == 2
    assert result[0] == Range(1, 5)
    assert result[1] == Range(10, 15)


def test_merges_and_keeps_separate_groups():
    # Arrange
    ranges = [
        Range(1, 3),
        Range(5, 7),
        Range(2, 4),
        Range(10, 15),
    ]

    # Act
    result = merge_ranges(ranges)

    # Assert
    assert len(result) == 3
    assert result[0] == Range(1, 4)
    assert result[1] == Range(5, 7)
    assert result[2] == Range(10, 15)


# Edge cases
def test_returns_empty_list_for_empty_input():
    # Arrange
    ranges = []

    # Act
    result = merge_ranges(ranges)

    # Assert
    assert result == []


def test_returns_single_range_unchanged():
    # Arrange
    ranges = [Range(1, 5)]

    # Act
    result = merge_ranges(ranges)

    # Assert
    assert len(result) == 1
    assert result[0] == Range(1, 5)


def test_handles_unsorted_input():
    # Arrange
    ranges = [
        Range(10, 15),
        Range(1, 5),
        Range(3, 7),
    ]

    # Act
    result = merge_ranges(ranges)

    # Assert
    assert len(result) == 2
    assert result[0] == Range(1, 7)
    assert result[1] == Range(10, 15)


def test_merges_fully_contained_range():
    # Arrange
    ranges = [
        Range(1, 10),
        Range(3, 5),
    ]

    # Act
    result = merge_ranges(ranges)

    # Assert
    assert len(result) == 1
    assert result[0] == Range(1, 10)


def test_merges_duplicate_ranges():
    # Arrange
    ranges = [
        Range(1, 5),
        Range(1, 5),
        Range(1, 5),
    ]

    # Act
    result = merge_ranges(ranges)

    # Assert
    assert len(result) == 1
    assert result[0] == Range(1, 5)


def test_handles_minimal_length_ranges():
    # Arrange
    ranges = [
        Range(1, 2),
        Range(2, 3),
        Range(3, 4),
    ]

    # Act
    result = merge_ranges(ranges)

    # Assert
    assert len(result) == 1
    assert result[0] == Range(1, 4)


def test_handles_chain_of_overlapping_ranges():
    # Arrange
    ranges = [
        Range(1, 5),
        Range(4, 8),
        Range(7, 10),
        Range(9, 12),
    ]

    # Act
    result = merge_ranges(ranges)

    # Assert
    assert len(result) == 1
    assert result[0] == Range(1, 12)


# Boundary tests
@pytest.mark.parametrize(
    ("ranges", "expected_count", "expected_ranges"),
    [
        # One gap between ranges
        (
            [Range(1, 3), Range(4, 6)],
            2,
            [Range(1, 3), Range(4, 6)],
        ),
        # Just adjacent
        (
            [Range(1, 3), Range(3, 6)],
            1,
            [Range(1, 6)],
        ),
        # Just overlapping by 1
        (
            [Range(1, 4), Range(3, 6)],
            1,
            [Range(1, 6)],
        ),
    ],
)
def test_boundary_conditions(ranges, expected_count, expected_ranges):
    # Act
    result = merge_ranges(ranges)

    # Assert
    assert len(result) == expected_count
    assert result == expected_ranges


def test_handles_large_number_of_ranges():
    # Arrange
    # Создаем 100 диапазонов, каждый перекрывается с предыдущим на 1
    ranges = [Range(i, i + 3) for i in range(0, 200, 2)]

    # Act
    result = merge_ranges(ranges)

    # Assert
    assert len(result) == 1
    assert result[0] == Range(0, 201)


def test_complex_scenario_with_multiple_groups():
    # Arrange
    ranges = [
        Range(1, 5),
        Range(3, 7),
        Range(10, 12),
        Range(11, 15),
        Range(20, 25),
        Range(30, 35),
        Range(32, 38),
        Range(37, 40),
    ]

    # Act
    result = merge_ranges(ranges)

    # Assert
    assert len(result) == 4
    assert result[0] == Range(1, 7)
    assert result[1] == Range(10, 15)
    assert result[2] == Range(20, 25)
    assert result[3] == Range(30, 40)


def test_preserves_range_with_no_overlap_before_and_after():
    # Arrange
    ranges = [
        Range(1, 5),
        Range(10, 15),
        Range(20, 25),
    ]

    # Act
    result = merge_ranges(ranges)

    # Assert
    assert len(result) == 3
    assert result == ranges


def test_merges_when_second_range_extends_first():
    # Arrange
    ranges = [
        Range(1, 10),
        Range(5, 15),
    ]

    # Act
    result = merge_ranges(ranges)

    # Assert
    assert len(result) == 1
    assert result[0] == Range(1, 15)


def test_merges_when_second_range_is_fully_contained():
    # Arrange
    ranges = [
        Range(1, 20),
        Range(5, 10),
    ]

    # Act
    result = merge_ranges(ranges)

    # Assert
    assert len(result) == 1
    assert result[0] == Range(1, 20)
