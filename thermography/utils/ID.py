__initial_value = -1


def reset_id(value: int = 0):
    """
    Resets the id of the ID generator.

    :param value: New start value of the ID generator.
    """
    global __initial_value
    __initial_value = value - 1


def next_id():
    """
    Generates the next unique id.

    :return: The generated ID
    """
    global __initial_value
    __initial_value += 1
    return __initial_value
