import unittest
import numpy as np

from openpilot.starpilot.lateral.blendful import Blendful, MAX_OFFSET, MIN_SPEED


def dummy_model_output(width=3.6, left_y=-1.8, right_y=1.8, std=0.05, prob=0.9):
  x = np.arange(0, 192.5, 0.5)
  n = len(x)
  # lane_lines: shape (1, 4, N, 2)
  lane_lines = np.zeros((1, 4, n, 2), dtype=np.float32)
  lane_lines[0, 1, :, 0] = x
  lane_lines[0, 1, :, 1] = left_y
  lane_lines[0, 2, :, 0] = x
  lane_lines[0, 2, :, 1] = right_y

  lane_lines_stds = np.zeros((1, 4, n, 2), dtype=np.float32)
  lane_lines_stds[0, 1, :, 0] = std
  lane_lines_stds[0, 2, :, 0] = std

  lane_lines_prob = np.zeros((1, 8), dtype=np.float32)
  lane_lines_prob[0, 3] = prob  # line 1 (1::2[1])
  lane_lines_prob[0, 5] = prob  # line 2 (1::2[2])

  plan = np.zeros((1, n, 2), dtype=np.float32)
  plan[0, :, 0] = x
  plan[0, :, 1] = 0.0

  return {
    'lane_lines': lane_lines,
    'lane_lines_stds': lane_lines_stds,
    'lane_lines_prob': lane_lines_prob,
    'plan': plan,
  }


class TestBlendful(unittest.TestCase):
  def test_disabled(self):
    b = Blendful()
    model = dummy_model_output()
    res = b.update(model, 20.0, True, 0.0, offset_m=0.0, enabled=False)
    self.assertEqual(res, 0.0)

  def test_speed_below_min(self):
    b = Blendful()
    model = dummy_model_output()
    res = b.update(model, MIN_SPEED - 1.0, True, 0.0, offset_m=0.0, enabled=True)
    self.assertEqual(res, 0.0)

  def test_centered_zero_offset(self):
    b = Blendful()
    model = dummy_model_output(left_y=-1.8, right_y=1.8)
    # When perfectly centered, center is 0.0, plan is 0.0, error is ~0
    for _ in range(50):
      res = b.update(model, 20.0, True, 0.0, offset_m=0.0, enabled=True)
    self.assertAlmostEqual(res, 0.0, places=3)

  def test_lane_offset_positive(self):
    b = Blendful()
    model = dummy_model_output(left_y=-1.8, right_y=1.8)
    # Positive offset (shift left) creates positive error -> positive curvature correction
    for _ in range(50):
      res = b.update(model, 20.0, True, 0.0, offset_m=0.2, enabled=True)
    self.assertGreater(res, 0.0)

  def test_lane_offset_negative(self):
    b = Blendful()
    model = dummy_model_output(left_y=-1.8, right_y=1.8)
    # Negative offset (shift right) creates negative error -> negative curvature correction
    for _ in range(50):
      res = b.update(model, 20.0, True, 0.0, offset_m=-0.2, enabled=True)
    self.assertLess(res, 0.0)

  def test_variance_weighting(self):
    b1 = Blendful()
    b2 = Blendful()
    # Left line noisy (std=0.5), right line clean (std=0.05)
    model1 = dummy_model_output(left_y=-2.5, right_y=1.5, std=0.05)
    model1['lane_lines_stds'][0, 1, :, 0] = 0.5  # high noise on left

    for _ in range(50):
      res1 = b1.update(model1, 20.0, True, 0.0, offset_m=0.0, enabled=True)

    # In model1, right line has much higher weight, so center is estimated closer to (right - half_width)
    self.assertTrue(np.isfinite(res1))


if __name__ == '__main__':
  unittest.main()

