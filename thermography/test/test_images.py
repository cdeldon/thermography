import numpy as np
import unittest

from thermography.utils.images import scale_image


class TestImageUtils(unittest.TestCase):
    def setUp(self):
        self.gray_image = np.ndarray((100, 200), dtype=np.uint8)
        self.rgb_image = np.ndarray((100, 200, 3), dtype=np.uint8)

    def test_scale_image_gray_identity(self):
        scaled = scale_image(self.gray_image, 1)
        self.assertEqual(self.gray_image.shape, scaled.shape)

    def test_scale_image_gray(self):
        larger = scale_image(self.gray_image, 1.5)
        self.assertEqual((150, 300), larger.shape)

        smaller = scale_image(self.gray_image, 0.2)
        self.assertEqual((20, 40), smaller.shape)

    def test_scale_image_rgb_identity(self):
        scaled = scale_image(self.rgb_image, 1)
        self.assertEqual(self.rgb_image.shape, scaled.shape)

    def test_scale_image_rgb(self):
        larger = scale_image(self.rgb_image, 1.5)
        self.assertEqual((150, 300, 3), larger.shape)

        smaller = scale_image(self.rgb_image, 0.2)
        self.assertEqual((20, 40, 3), smaller.shape)


if __name__ == '__main__':
    unittest.main()
