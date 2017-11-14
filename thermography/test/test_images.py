"""This module implements the tests of the functions in :mod:`geometry <thermography.utils.images>` package."""

import unittest

import numpy as np

from thermography.utils.images import *


class TestImageUtils(unittest.TestCase):
    def setUp(self):
        """Initializes the common elements for the tests."""
        self.gray_image = np.ndarray((100, 200), dtype=np.uint8)
        self.rgb_image = np.ndarray((100, 200, 3), dtype=np.uint8)

    def test_scale_image_gray_identity(self):
        """Tests the shape of the input gray_scale image after being scaled by the identity.
        """
        scaled = scale_image(self.gray_image, 1)
        self.assertEqual(self.gray_image.shape, scaled.shape)

    def test_scale_image_gray(self):
        """Tests the shape of the input gray_scale image after being scaled.
        """
        larger = scale_image(self.gray_image, 1.5)
        self.assertEqual((150, 300), larger.shape)

        smaller = scale_image(self.gray_image, 0.2)
        self.assertEqual((20, 40), smaller.shape)

    def test_scale_image_rgb_identity(self):
        """Tests the shape of the input rgb image after being scaled by the identity.
        """
        scaled = scale_image(self.rgb_image, 1)
        self.assertEqual(self.rgb_image.shape, scaled.shape)

    def test_scale_image_rgb(self):
        """Tests the shape of the input rgb image after being scaled.
        """
        larger = scale_image(self.rgb_image, 1.5)
        self.assertEqual((150, 300, 3), larger.shape)

        smaller = scale_image(self.rgb_image, 0.2)
        self.assertEqual((20, 40, 3), smaller.shape)


if __name__ == '__main__':
    unittest.main()
