import numpy as np
from types import SimpleNamespace


def are_equal(struct1, struct2, check_types=True):
    """
    Compare two potentially nested structures containing Python objects, including numpy arrays and namespaces.

    Parameters
    ----------
    - struct1, struct2: The structures to compare. These could be dicts, lists, tuples, SimpleNamespace, or primitive types.
    - check_types: If True, it will check whether the types of struct1 and struct2 match.

    Returns
    -------
    - bool: True if both structures are equal, False otherwise.
    """
    # Case 1: If both are numpy arrays, use np.array_equal to compare
    if isinstance(struct1, np.ndarray) and isinstance(struct2, np.ndarray):
        return np.array_equal(struct1, struct2)

    # Case 2: If both are SimpleNamespace, compare their __dict__ attributes
    elif isinstance(struct1, SimpleNamespace) and isinstance(struct2, SimpleNamespace):
        return are_equal(vars(struct1), vars(struct2), check_types)

    # Case 3: If both are dictionaries, compare their keys and values
    elif isinstance(struct1, dict) and isinstance(struct2, dict):
        if struct1.keys() != struct2.keys():
            return False
        return all(
            are_equal(struct1[key], struct2[key], check_types) for key in struct1
        )

    # Case 4: If both are lists or tuples, compare their elements
    elif isinstance(struct1, (list, tuple)) and isinstance(struct2, (list, tuple)):
        if len(struct1) != len(struct2):
            return False
        return all(
            are_equal(item1, item2, check_types)
            for item1, item2 in zip(struct1, struct2)
        )

    # Case 5: If both are of the same basic type (int, float, str, etc.), compare directly
    elif check_types and type(struct1) is not type(struct2):
        return False
    else:
        return struct1 == struct2


# Example usage
if __name__ == "__main__":
    ns1 = SimpleNamespace(a=1, b=np.array([1, 2, 3]), c=SimpleNamespace(d=4))
    ns2 = SimpleNamespace(a=1, b=np.array([1, 2, 3]), c=SimpleNamespace(d=4))

    print(are_equal(ns1, ns2))  # Output: True
