import numpy as np


def angle(pt1, pt2):
    a = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
    if a > np.pi:
        a -= np.pi
    elif a < 0:
        a += np.pi
    return a
