"""This module implements a simple unique-ID generator. The generated IDs follow the natural numbers order.

:Example:

    .. code-block:: python

        first_id = next_id() # first_id = 0
        second_id = next_id() # second_id = 1

        reset_id()
        third_id = next_id() # third_id = 0

        reset_id(10)
        fourth_id = next_id() # fourth_id = 10
"""

__initial_value = -1


def reset_id(value: int = 0):
    """Resets the id of the ID generator.

    :param value: New start value of the ID generator.
    """
    global __initial_value
    __initial_value = value - 1


def next_id():
    """Generates the next unique id.

    :return: The generated ID
    """
    global __initial_value
    __initial_value += 1
    return __initial_value
