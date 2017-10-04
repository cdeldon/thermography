import cv2
import numpy as np

from thermography.utils import random_color

__all__ = ["draw_intersections", "draw_rectangles", "draw_segments",
           "rotate_image",
           "scale_image"]


def draw_intersections(intersections: list, base_image: np.ndarray, windows_name: str):
    """
    Draws the intersections contained in the first parameter onto the base image passed as second parameter and displays
    the image using the third parameter as title.
    :param intersections: List of intersection coordinates.
    :param base_image: Base image over which to render the intersections.
    :param windows_name: Title to give to the rendered image.
    """
    for intersection in intersections:
        cv2.circle(base_image, (int(intersection[0]), int(intersection[1])), 2, (0, 0, 255), 2, cv2.LINE_4)

    cv2.imshow(windows_name, base_image)


def draw_segments(segments: list, base_image: np.ndarray, windows_name: str, render_indices: bool = True,
                  colors: list = None):
    """
    Draws the segments contained in the first parameter onto the base image passed as second parameter and displays
    the image using the third parameter as title. The indices associated to the segments are rendere on the image
    depending on 'render_indices'.
    A list of colors can be passed as argument to specify the colors to be used for different segment clusters.
    :param segments: List of segment clusters.
    :param base_image: Base image over which to render the segments.
    :param windows_name: Title to give to the rendered image.
    :param render_indices: Boolean flag indicating whether to render the segment indices or not.
    :param colors: Color list to be used for segment rendering, such that segments belonging to the same cluster are of
    the same color.
    """
    if colors is None:
        # Fix colors for first two clusters, choose the next randomly.
        colors = [(29, 247, 240), (255, 180, 50)]
        for cluster_number in range(2, len(segments)):
            colors.append(random_color())

    for cluster, color in zip(segments, colors):
        for segment_index, segment in enumerate(cluster):
            cv2.line(img=base_image, pt1=(segment[0], segment[1]), pt2=(segment[2], segment[3]),
                     color=color, thickness=1, lineType=cv2.LINE_AA)
            if render_indices:
                cv2.putText(base_image, str(segment_index), (segment[0], segment[1]), cv2.FONT_HERSHEY_PLAIN, 0.8,
                            (255, 255, 255), 1)

    cv2.imshow(windows_name, base_image)


def draw_rectangles(rectangles: list, base_image: np.ndarray, windows_name: str):
    """
    Draws the rectangles contained in the first parameter onto the base image passed as second parameter and displays
    the image using the third parameter as title.
    :param rectangles: List of rectangles.
    :param base_image: Base image over which to render the rectangles.
    :param windows_name: Title to give to the rendered image.
    """
    for rectangle in rectangles:
        cv2.polylines(base_image, np.int32([rectangle]), True, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow(windows_name, base_image)


def rotate_image(image: np.ndarray, a: float):
    """
    Rotates the input image by a radiants in counter-clock-wise direction.
    :param image: Image to be rotated.
    :param a: Rotation angle expressed in radiants.
    :return: Rotated version of the input image.
    """
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, a / np.pi * 180, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def scale_image(input_image: np.ndarray, s: float):
    """
    Scales an input image by the value passed as parameter.
    :param input_image: Image to be scaled.
    :param s: Scalar value to be applied to the scaling procedure.
    :return: A copy of the input image scaled by the passed parameter.
    """
    return cv2.resize(src=input_image, dsize=(0, 0), fx=s, fy=s)
