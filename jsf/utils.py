from typing import List, Union
from jsf.types import Time


def array_subtract_ab(array_a: List[float], array_b: List[float]) -> List[float]:
    """Subtract array B from array A element-wise."""
    return [a - b for a, b in zip(array_a, array_b)]


def array_plus_ab(array_a: List[float], array_b: List[float]) -> List[float]:
    """Add array A and array B element-wise."""
    return [a + b for a, b in zip(array_a, array_b)]


def matrix_subtract_ab(matrix_a: List[List[float]], matrix_b: List[List[float]]) -> List[List[float]]:
    """Subtract matrix B from matrix A element-wise."""
    return [[a - b for a, b in zip(row1, row2)] for row1, row2 in zip(matrix_a, matrix_b)]


def matrix_plus_ab(matrix_a: List[List[float]], matrix_b: List[List[float]]) -> List[List[float]]:
    """Add matrix A and matrix B element-wise."""
    return [[a + b for a, b in zip(row1, row2)] for row1, row2 in zip(matrix_a, matrix_b)]


def num_non_zero(array: Union[List[float], List[int], List[Time]]) -> int:
    """Count non-zero elements in array."""
    return sum(1 for element in array if element != 0)


def matrix_dot_array(matrix: List[List[float]], array: List[float]) -> List[float]:
    """Compute matrix-vector product."""
    return [sum(row[i] * array[i] for i in range(len(array))) for row in matrix]


def array_multiply_ab(array_a: List[float], array_b: List[float]) -> List[float]:
    """Multiply array A and array B element-wise."""
    return [a * b for a, b in zip(array_a, array_b)]