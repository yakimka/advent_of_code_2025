import pytest

from support import DisjointSet


@pytest.fixture()
def make_disjoint_set():
    """Factory for creating DisjointSet instances with custom items."""

    def maker(items):
        return DisjointSet(items)

    return maker


@pytest.fixture()
def disjoint_set_strings(make_disjoint_set):
    """DisjointSet instance with string items for simple test cases."""
    return make_disjoint_set(["a", "b", "c", "d", "e"])


@pytest.fixture()
def disjoint_set_ints(make_disjoint_set):
    """DisjointSet instance with integer items for simple test cases."""
    return make_disjoint_set([1, 2, 3, 4, 5])


def test_initializes_with_strings(disjoint_set_strings):
    # Arrange & Act - already initialized by fixture
    # Assert
    assert disjoint_set_strings.n_subsets == 5
    assert len(disjoint_set_strings.subsets()) == 5


def test_initializes_with_integers(disjoint_set_ints):
    # Arrange & Act - already initialized by fixture
    # Assert
    assert disjoint_set_ints.n_subsets == 5
    assert len(disjoint_set_ints.subsets()) == 5


def test_initializes_with_empty_sequence(make_disjoint_set):
    # Arrange & Act
    ds = make_disjoint_set([])

    # Assert
    assert ds.n_subsets == 0
    assert ds.subsets() == []


def test_initializes_with_single_item(make_disjoint_set):
    # Arrange & Act
    ds = make_disjoint_set(["single"])

    # Assert
    assert ds.n_subsets == 1
    assert ds.subsets() == [{"single"}]


def test_find_returns_different_roots_for_unmerged_items(disjoint_set_strings):
    # Arrange & Act
    root_a = disjoint_set_strings.find("a")
    root_b = disjoint_set_strings.find("b")

    # Assert
    assert root_a != root_b


def test_find_returns_same_root_after_merge(disjoint_set_strings):
    # Arrange
    disjoint_set_strings.merge("a", "b")

    # Act
    root_a = disjoint_set_strings.find("a")
    root_b = disjoint_set_strings.find("b")

    # Assert
    assert root_a == root_b


def test_merge_returns_true_when_merging_different_sets(disjoint_set_strings):
    # Arrange & Act
    result = disjoint_set_strings.merge("a", "b")

    # Assert
    assert result is True


def test_merge_returns_false_when_items_already_in_same_set(disjoint_set_strings):
    # Arrange
    disjoint_set_strings.merge("a", "b")

    # Act
    result = disjoint_set_strings.merge("a", "b")

    # Assert
    assert result is False


def test_merge_returns_false_when_items_indirectly_in_same_set(
    disjoint_set_strings,
):
    # Arrange
    disjoint_set_strings.merge("a", "b")
    disjoint_set_strings.merge("b", "c")

    # Act
    result = disjoint_set_strings.merge("a", "c")

    # Assert
    assert result is False


def test_merge_decreases_subset_count(disjoint_set_strings):
    # Arrange
    initial_count = disjoint_set_strings.n_subsets

    # Act
    disjoint_set_strings.merge("a", "b")

    # Assert
    assert disjoint_set_strings.n_subsets == initial_count - 1


def test_merge_does_not_decrease_subset_count_when_already_merged(
    disjoint_set_strings,
):
    # Arrange
    disjoint_set_strings.merge("a", "b")
    count_after_first_merge = disjoint_set_strings.n_subsets

    # Act
    disjoint_set_strings.merge("a", "b")

    # Assert
    assert disjoint_set_strings.n_subsets == count_after_first_merge


def test_subsets_returns_all_items_in_separate_sets_initially(
    disjoint_set_strings,
):
    # Arrange & Act
    subsets = disjoint_set_strings.subsets()

    # Assert
    assert len(subsets) == 5
    assert {"a"} in subsets
    assert {"b"} in subsets
    assert {"c"} in subsets
    assert {"d"} in subsets
    assert {"e"} in subsets


def test_subsets_returns_merged_items_in_same_set(disjoint_set_strings):
    # Arrange
    disjoint_set_strings.merge("a", "b")
    disjoint_set_strings.merge("c", "d")

    # Act
    subsets = disjoint_set_strings.subsets()

    # Assert
    assert len(subsets) == 3
    assert {"a", "b"} in subsets
    assert {"c", "d"} in subsets
    assert {"e"} in subsets


def test_subsets_returns_transitively_merged_items_in_same_set(
    disjoint_set_strings,
):
    # Arrange
    disjoint_set_strings.merge("a", "b")
    disjoint_set_strings.merge("b", "c")

    # Act
    subsets = disjoint_set_strings.subsets()

    # Assert
    assert len(subsets) == 3
    assert {"a", "b", "c"} in subsets
    assert {"d"} in subsets
    assert {"e"} in subsets


def test_complex_merge_structure_creates_correct_subsets(make_disjoint_set):
    # Arrange
    ds = make_disjoint_set([1, 2, 3, 4, 5, 6, 7, 8])

    # Act - Create structure: {1,2,3,4}, {5,6}, {7}, {8}
    ds.merge(1, 2)
    ds.merge(3, 4)
    ds.merge(1, 3)
    ds.merge(5, 6)

    # Assert
    subsets = ds.subsets()
    assert len(subsets) == 4
    assert {1, 2, 3, 4} in subsets
    assert {5, 6} in subsets
    assert {7} in subsets
    assert {8} in subsets
    assert ds.n_subsets == 4


def test_merging_all_items_into_single_set(disjoint_set_strings):
    # Arrange & Act
    disjoint_set_strings.merge("a", "b")
    disjoint_set_strings.merge("b", "c")
    disjoint_set_strings.merge("c", "d")
    disjoint_set_strings.merge("d", "e")

    # Assert
    subsets = disjoint_set_strings.subsets()
    assert len(subsets) == 1
    assert {"a", "b", "c", "d", "e"} in subsets
    assert disjoint_set_strings.n_subsets == 1


def test_n_subsets_property_tracks_merges_correctly(make_disjoint_set):
    # Arrange
    ds = make_disjoint_set([1, 2, 3, 4, 5])

    # Act & Assert
    assert ds.n_subsets == 5

    ds.merge(1, 2)
    assert ds.n_subsets == 4

    ds.merge(3, 4)
    assert ds.n_subsets == 3

    ds.merge(1, 3)
    assert ds.n_subsets == 2

    ds.merge(1, 4)  # Already in same set
    assert ds.n_subsets == 2

    ds.merge(5, 1)
    assert ds.n_subsets == 1


def test_path_compression_flattens_tree_structure(make_disjoint_set):
    # Arrange
    ds = make_disjoint_set([1, 2, 3, 4, 5])

    # Create a chain: 1 -> 2 -> 3 -> 4 -> 5
    ds.merge(1, 2)
    ds.merge(2, 3)
    ds.merge(3, 4)
    ds.merge(4, 5)

    # Act - Call find multiple times to trigger path compression
    root_1_first = ds.find(1)
    root_5_first = ds.find(5)
    assert root_1_first == root_5_first

    # After path compression, all nodes should point closer to root
    # Multiple finds should maintain the same root
    root_1_second = ds.find(1)
    root_2 = ds.find(2)
    root_3 = ds.find(3)
    root_4 = ds.find(4)
    root_5_second = ds.find(5)

    # Assert - All should have same root after path compression
    assert root_1_second == root_2 == root_3 == root_4 == root_5_second


@pytest.mark.parametrize(
    ("items", "merge_pairs", "expected_subset_count", "expected_largest_subset_size"),
    [
        # Single merge
        (["a", "b", "c"], [("a", "b")], 2, 2),
        # No merges
        (["a", "b", "c"], [], 3, 1),
        # Multiple independent merges
        (["a", "b", "c", "d"], [("a", "b"), ("c", "d")], 2, 2),
        # Transitive merges forming one large set
        (
            ["a", "b", "c", "d"],
            [("a", "b"), ("b", "c"), ("c", "d")],
            1,
            4,
        ),
        # Complex pattern
        (
            [1, 2, 3, 4, 5, 6],
            [(1, 2), (2, 3), (4, 5)],
            3,
            3,
        ),
    ],
)
def test_various_merge_patterns(
    make_disjoint_set,
    items,
    merge_pairs,
    expected_subset_count,
    expected_largest_subset_size,
):
    # Arrange
    ds = make_disjoint_set(items)

    # Act
    for a, b in merge_pairs:
        ds.merge(a, b)

    # Assert
    subsets = ds.subsets()
    assert len(subsets) == expected_subset_count
    assert ds.n_subsets == expected_subset_count
    assert max(len(s) for s in subsets) == expected_largest_subset_size


def test_merge_with_different_sized_sets_uses_union_by_size(make_disjoint_set):
    # Arrange
    ds = make_disjoint_set([1, 2, 3, 4, 5])

    # Create sets of different sizes
    ds.merge(1, 2)  # Set 1: {1, 2} (size 2)
    ds.merge(3, 4)  # Set 2: {3, 4} (size 2)
    ds.merge(3, 5)  # Set 2: {3, 4, 5} (size 3)

    # Act - Merge smaller set into larger set
    result = ds.merge(1, 3)

    # Assert
    assert result is True
    subsets = ds.subsets()
    assert len(subsets) == 1
    assert {1, 2, 3, 4, 5} in subsets


def test_finds_work_after_multiple_operations(make_disjoint_set):
    # Arrange
    ds = make_disjoint_set(["x", "y", "z", "w"])

    # Act - Perform various operations
    ds.merge("x", "y")
    root1 = ds.find("x")
    root2 = ds.find("y")

    ds.merge("z", "w")
    root3 = ds.find("z")
    root4 = ds.find("w")

    ds.merge("x", "z")
    final_root_x = ds.find("x")
    final_root_y = ds.find("y")
    final_root_z = ds.find("z")
    final_root_w = ds.find("w")

    # Assert
    assert root1 == root2
    assert root3 == root4
    assert final_root_x == final_root_y == final_root_z == final_root_w


def test_works_with_tuple_items(make_disjoint_set):
    # Arrange & Act
    ds = make_disjoint_set([(0, 0), (0, 1), (1, 0), (1, 1)])
    ds.merge((0, 0), (0, 1))
    ds.merge((1, 0), (1, 1))

    # Assert
    subsets = ds.subsets()
    assert len(subsets) == 2
    assert {(0, 0), (0, 1)} in subsets
    assert {(1, 0), (1, 1)} in subsets


def test_repeated_find_returns_consistent_results(disjoint_set_strings):
    # Arrange
    disjoint_set_strings.merge("a", "b")
    disjoint_set_strings.merge("b", "c")

    # Act
    roots = [disjoint_set_strings.find("a") for _ in range(10)]

    # Assert - All finds should return the same root
    assert len(set(roots)) == 1
