"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import numpy as np

from openpilot.sunnypilot.selfdrive.controls.lib.dec.dec import DynamicExperimentalController
from openpilot.sunnypilot.selfdrive.controls.lib.dec.constants import WMACConstants

_STOP_SHAPE_A_FLOOR = WMACConstants.STOP_SHAPE_A_FLOOR


def _dec(enabled=True, lead=False, slow_down=True, v_ego=12.0, accel_min=-2.5):
  # apply_stop_shaping is pure on stored DEC state, so build a bare instance and set it.
  # Gate is _active (experimental + DEC enabled), not the raw _enabled param.
  c = DynamicExperimentalController.__new__(DynamicExperimentalController)
  c._active = enabled
  c._has_lead_filtered = lead
  c._has_slow_down = slow_down
  c._v_ego = v_ego
  c._model_accel_min = accel_min
  return c


class TestDecStopShaping:
  def test_disabled_passthrough(self):
    assert _dec(enabled=False).apply_stop_shaping(-1.5) == -1.5

  def test_acts_even_with_radar_lead(self):
    # model-accel front-load is no longer gated on no-lead
    assert _dec(lead=True, accel_min=-2.5).apply_stop_shaping(-1.0) == _STOP_SHAPE_A_FLOOR

  def test_no_slow_down_passthrough(self):
    assert _dec(slow_down=False).apply_stop_shaping(-1.5) == -1.5

  def test_not_braking_passthrough(self):
    assert _dec(accel_min=-2.5).apply_stop_shaping(0.2) == 0.2

  def test_speed_out_of_band_passthrough(self):
    assert _dec(v_ego=30.0).apply_stop_shaping(-1.5) == -1.5
    assert _dec(v_ego=1.0).apply_stop_shaping(-1.5) == -1.5

  def test_small_margin_passthrough(self):
    # predicted decel barely harder than commanded -> ignore
    assert _dec(accel_min=-1.2).apply_stop_shaping(-1.0) == -1.0

  def test_front_loads_to_floor(self):
    # model foresees -2.5, pulled forward but capped at the comfort floor
    assert _dec(accel_min=-2.5).apply_stop_shaping(-1.0) == _STOP_SHAPE_A_FLOOR

  def test_floor_caps_deep_prediction(self):
    assert _dec(accel_min=-3.5).apply_stop_shaping(-0.6) == _STOP_SHAPE_A_FLOOR

  def test_never_reduces_requested_braking(self):
    # model already commanding harder than the floor -> keep it, don't soften
    assert _dec(accel_min=-3.5).apply_stop_shaping(-3.0) == -3.0

  def test_monotonic_never_less_braking(self):
    c = _dec(accel_min=-2.5)
    for a_target in np.linspace(-0.6, -3.0, 25):
      assert c.apply_stop_shaping(float(a_target)) <= float(a_target) + 1e-9
