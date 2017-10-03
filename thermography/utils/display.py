import numpy as np

__all__ = ["random_color"]


def random_color() -> tuple:
    """
    Generates a random RGB color in [0, 255]^3
    :return: A randomly generated color defined as a triplet of RGB values.
    """
    c = np.random.randint(0, 255, 3)
    return int(c[0]), int(c[1]), int(c[2])
