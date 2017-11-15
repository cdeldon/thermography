"""This module contains multiple utility functions which can be used to display intermediate representations computed by the :class:`ThermoApp <thermography.thermo_app.ThermoApp>` class."""

import cv2
import numpy as np

__all__ = ["draw_intersections", "draw_motion", "draw_rectangles", "draw_segments",
           "random_color", "color_from_probabilities"]


def draw_intersections(intersections: list, base_image: np.ndarray, windows_name: str):
    """Draws the intersections contained in the first parameter onto the base image passed as second parameter and displays the image using the third parameter as title.

    :param intersections: List of intersection coordinates.
    :param base_image: Base image over which to render the intersections.
    :param windows_name: Title to give to the rendered image.
    """
    mean_color = np.mean(base_image, axis=(0, 1))
    if mean_color[0] == mean_color[1] == mean_color[2]:
        mean_color = np.array([255, 255, 0])
    opposite_color = np.array([255, 255, 255]) - mean_color
    opposite_color = (int(opposite_color[0]), int(opposite_color[1]), int(opposite_color[2]))
    for intersection in intersections:
        cv2.circle(img=base_image, center=(int(intersection[0]), int(intersection[1])), radius=2, color=opposite_color,
                   thickness=3, lineType=cv2.LINE_4)

    cv2.imshow(windows_name, base_image)


def draw_motion(flow: np.ndarray, base_image: np.ndarray, windows_name: str, draw_mean_motion: bool = True, nums=10):
    """Draws the motion flow contained in the first parameter onto the base image passed as second argument and displays the image using the third argument as title.

    :param flow: Numpy array of flow computed by the motion estimator.
    :param base_image: Base image over which to render the intersection. Note that this image must have the dimensions used for the flow computation.
    :param windows_name: Title to give to the rendered image.
    :param draw_mean_motion: Boolean flag indicating whether to render also the mean motion estimate.
    :param nums: Number of samples in each dimension to be rendered.
    """
    if flow is not None:
        if len(base_image.shape) == 2:
            base_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGRA)
        h, w = base_image.shape[:2]
        step_y = h / nums
        step_x = w / nums
        y, x = np.mgrid[step_y / 2:h:step_y, step_x / 2:w:step_x].reshape(2, -1)
        x = x.astype(int)
        y = y.astype(int)
        fx, fy = flow[y, x].T * 5
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)

        cv2.polylines(base_image, lines, 0, (0, 255, 0))
        for (x1, y1), _ in lines:
            cv2.circle(base_image, (x1, y1), 1, (0, 255, 0), cv2.FILLED)

        if draw_mean_motion:
            mean_flow_x = -np.mean(fx) * 5
            mean_flow_y = -np.mean(fy) * 5
            center = (np.array([w, h]) * 0.5 + 0.5).astype(int)
            cv2.arrowedLine(base_image, (center[0], center[1]),
                            (int(mean_flow_x + center[0]), int(mean_flow_y + center[1])),
                            (0, 0, 255), 2, cv2.LINE_4)

    cv2.imshow(windows_name, base_image)


def draw_segments(segments: list, base_image: np.ndarray, windows_name: str, render_indices: bool = True,
                  colors: list = None):
    """Draws the segments contained in the first parameter onto the base image passed as second parameter.

    This function displays the image using the third parameter as title.
    The indices associated to the segments are rendered on the image depending on 'render_indices'.
    A list of colors can be passed as argument to specify the colors to be used for different segment clusters.

    :param segments: List of segment clusters.
    :param base_image: Base image over which to render the segments.
    :param windows_name: Title to give to the rendered image.
    :param render_indices: Boolean flag indicating whether to render the segment indices or not.
    :param colors: Color list to be used for segment rendering, such that segments belonging to the same cluster are of the same color.
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
    """Draws the rectangles contained in the first parameter onto the base image passed as second parameter.

    This function displays the image using the third parameter as title.

    :param rectangles: List of rectangles.
    :param base_image: Base image over which to render the rectangles.
    :param windows_name: Title to give to the rendered image.
    """
    mean_color = np.mean(base_image, axis=(0, 1))
    mask = np.zeros_like(base_image)
    if mean_color[0] == mean_color[1] == mean_color[2]:
        mean_color = np.array([255, 255, 0])
    opposite_color = np.array([255, 255, 255]) - mean_color
    opposite_color = (int(opposite_color[0]), int(opposite_color[1]), int(opposite_color[2]))
    for rectangle in rectangles:
        cv2.polylines(base_image, np.int32([rectangle]), True, opposite_color, 1, cv2.LINE_AA)
        cv2.fillConvexPoly(mask, np.int32([rectangle]), (255, 0, 0), cv2.LINE_4)

    cv2.addWeighted(base_image, 1, mask, 0.3, 0, base_image)

    cv2.imshow(windows_name, base_image)


def random_color() -> tuple:
    """Generates a random RGB color in [0, 255]^3

    :return: A randomly generated color defined as a triplet of RGB values.
    """
    c = np.random.randint(0, 255, 3)
    return int(c[0]), int(c[1]), int(c[2])


def color_from_probabilities(prob: np.ndarray) -> tuple:
    """Constructs a color tuple given the probability distribution prob.

    :param prob: A three dimensional numpy array containing class probabilities.
    :return: The color associated to the probability distribution.
    """
    color = np.diag(prob).dot(np.ones(shape=[3, 1]) * 255.0)
    return (int(color[2]), int(color[0]), int(color[1]))
