"""Variance-weighted lane centering with user-controllable offset.

Adapted from zoompilot's Blendful algorithm: the center estimate weights each
lane line inversely by its variance (noisy line → less influence), and the model
plan trajectory is blended in during curves.  The user's LaneCenterOffset is
applied to the weighted center, clamped so the vehicle stays within body
clearance of both lane lines.

This module neither reads settings nor sends commands; it receives model data
and returns a curvature correction like the existing raw_correction() API.
"""

import math

import numpy as np

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.common.scalar_clip import clip_scalar


# Lane geometry
LANE_WIDTH_DEFAULT = 3.6  # m
LANE_WIDTH_CLIP = (2.8, 4.5)  # m

# Speed and lookahead
LOOKAHEAD_TIME = 1.5  # s
MIN_LOOKAHEAD = 10.0  # m
MIN_SPEED = 5.0  # m/s

# Line quality thresholds
LINE_STD = (0.2, 0.6)  # maps line std → quality 1.0…0.0
POINT_STD = 0.1  # regularization for weighted least-squares

# Obstacle yielding
YIELD = (0.2, 0.45)  # m — swerve thresholds for trust reduction

# Offset limits (must agree with LaneCenterOffset param range)
MAX_OFFSET = 0.3  # m
MIN_CENTER_TO_LINE = 1.1  # m — minimum distance from center target to nearest line

# Model constants — the X indices for the lane line and plan arrays
from openpilot.selfdrive.modeld.constants import ModelConstants

X_IDXS = np.array(ModelConstants.X_IDXS)


class Blendful:
  """Variance-weighted lane centering with user offset and obstacle yielding.

  Call update() every model tick with the model output dict, vehicle speed,
  lateral active state, model curvature, user offset, and enabled flag.
  Returns a curvature correction (delta-kappa) to add to the base curvature.
  """

  def __init__(self):
    self.lane_width = FirstOrderFilter(LANE_WIDTH_DEFAULT, 10.0, DT_MDL, initialized=False)
    self.line_gone = FirstOrderFilter(np.zeros(2), 2.0, DT_MDL)
    self.line_probs_slow = FirstOrderFilter(np.zeros(2), 3.0, DT_MDL)
    self.line_probs = FirstOrderFilter(np.zeros(2), 0.5, DT_MDL)
    self.usual_offset = FirstOrderFilter(0.0, 10.0, DT_MDL, initialized=False)
    self.going_around = False
    self.since_turning = 0.0
    self.trust = 0.0
    self.line_trust = 0.0
    self.dk = 0.0  # delta kappa: lane center vs model plan curvature difference
    self.l_prob = 0.0
    self.r_prob = 0.0
    self.model_curvature = 0.0
    self.smooth = 0.0  # smooths the correction target below 0.53 lat accel
    self.curvature = 0.0

  def update(self, model_output: dict[str, np.ndarray], v_ego: float, active: bool,
             model_curvature: float, offset_m: float = 0.0, enabled: bool = True) -> float:
    """Compute a curvature correction for lane centering.

    Args:
      model_output: Dict with 'lane_lines', 'lane_lines_stds', 'lane_lines_prob', 'plan' arrays.
      v_ego: Vehicle speed in m/s.
      active: Whether lateral control is active.
      model_curvature: The model's current desired curvature (base).
      offset_m: User lane offset in meters (positive = left, from LaneCenterOffset param).
      enabled: Whether blendful lane centering is enabled.

    Returns:
      Curvature correction (delta-kappa) to add to the base curvature.
    """
    self.l_prob = self.r_prob = 0.0
    self.model_curvature = model_curvature
    if not enabled:
      if self.curvature or self.trust:
        self.__init__()
      return 0.0
    fading = not active or v_ego < MIN_SPEED
    if fading:
      self.trust = self.line_trust = 0.0
      self.usual_offset.initialized = False
      self.lane_width.initialized = False
      self.going_around = False
    offset = clip_scalar(offset_m, -MAX_OFFSET, MAX_OFFSET) if math.isfinite(offset_m) else 0.0
    target = 0.0 if fading else self._target(model_output, v_ego, model_curvature, offset)
    target = self.smooth = self.smooth + (target - self.smooth) * DT_MDL / (
      DT_MDL + np.interp(abs(model_curvature) * v_ego ** 2, [0.18, 0.53], [0.5, 0.0]))
    build = 0.5 / max(v_ego, 1.0) ** 2 * DT_MDL  # jerk over 1 second
    release = (0.5 if fading else 2.0) / max(v_ego, 1.0) ** 2 * DT_MDL
    k = self.curvature
    self.curvature = float(np.clip(target,
                                   max(k - release, -build) if k > 0 else k - build,
                                   min(k + release, build) if k < 0 else k + build))
    return self.curvature

  def _target(self, model_output: dict[str, np.ndarray], v_ego: float,
              model_curvature: float, offset: float) -> float:
    lookahead = max(v_ego * LOOKAHEAD_TIME, MIN_LOOKAHEAD)
    lines = model_output['lane_lines'][0, :, :, 0]
    stds = model_output['lane_lines_stds'][0, :, :, 0]

    lone_weight = np.interp(abs(model_curvature) * v_ego ** 2, [0.18, 0.53], [0.5, 1.0])
    probs = self.line_probs.update(model_output['lane_lines_prob'][0, 1::2][1:3])
    absent = (probs < 0.3) & (lone_weight > 0.5)
    gone = self.line_gone.update((model_output['lane_lines_prob'][0, 1::2][1:3] < 0.3).astype(float))
    missing = np.where(absent, 1.0, gone)
    dropping = bool(np.any((probs < 0.6 * self.line_probs_slow.update(
      model_output['lane_lines_prob'][0, 1::2][1:3])) & ~absent))
    l_prob = float(probs[0] * np.interp(np.interp(0.5 * lookahead, X_IDXS, stds[1]), LINE_STD, [1.0, 0.0]))
    r_prob = float(probs[1] * np.interp(np.interp(0.5 * lookahead, X_IDXS, stds[2]), LINE_STD, [1.0, 0.0]))
    self.l_prob, self.r_prob = l_prob, r_prob

    # Trust computation: how much to let lane lines influence steering
    if missing[1] > 0.8 and l_prob > 0.5:
      lane_trust = l_prob * lone_weight
    elif missing[0] > 0.8 and r_prob > 0.5:
      lane_trust = r_prob * lone_weight
    elif dropping:
      lane_trust = min(l_prob, r_prob)
    else:
      lane_trust = l_prob + r_prob - l_prob * r_prob
    if lone_weight > 0.5 and lane_trust < self.line_trust:
      lane_trust = self.line_trust + (lane_trust - self.line_trust) * DT_MDL / (lone_weight - 0.5 + DT_MDL)
    self.line_trust = lane_trust
    if lane_trust <= 0.0:
      self.trust = 0.0
      return 0.0
    if l_prob + r_prob <= 0.0:
      self.trust = min(self.trust, lane_trust)
      return float(self.trust * self.dk)

    # Windowed lane center estimation
    win = (X_IDXS >= 0.5 * lookahead) & (X_IDXS <= 1.5 * lookahead)
    x = X_IDXS[win]
    half_width = self.lane_width.x / 2
    if l_prob * r_prob > 0.5:
      width = np.interp(lookahead, X_IDXS, lines[2]) - np.interp(lookahead, X_IDXS, lines[1])
      self.lane_width.update(float(np.clip(width, *LANE_WIDTH_CLIP)))
      half_width = np.clip(lines[2, win] - lines[1, win], *LANE_WIDTH_CLIP) / 2

    # Variance-weighted center: noisier line contributes less
    l_var = stds[1, win] ** 2 / max(l_prob, 1e-4)
    r_var = stds[2, win] ** 2 / max(r_prob, 1e-4)
    center = (r_var * (lines[1, win] + half_width) + l_var * (lines[2, win] - half_width)) / (l_var + r_var)

    # Apply user offset, clamped to keep the target within safe bounds
    if offset != 0.0:
      if isinstance(half_width, np.ndarray):
        safe_offset = np.minimum(MAX_OFFSET, np.maximum(0.0, half_width - MIN_CENTER_TO_LINE))
      else:
        safe_offset = min(MAX_OFFSET, max(0.0, half_width - MIN_CENTER_TO_LINE))
      center = center + np.clip(offset, -safe_offset, safe_offset)

    weight = 1.0 / (l_var * r_var / (l_var + r_var) + POINT_STD ** 2)

    # Fit curvature to place the plant in center: error ~ dk·x²/2
    plan_x = np.interp(x, model_output['plan'][0, :, 0], model_output['plan'][0, :, 1])
    plan_share = np.interp(abs(model_curvature) * lookahead ** 2 / 2, [0.5, 2.0], [0.3, 0.5])
    error = np.clip(center - ((1 - plan_share) * model_curvature * x ** 2 / 2 + plan_share * plan_x), -1.0, 1.0)
    dk = 2.0 * np.sum(weight * error * x ** 2) / np.sum(weight * x ** 4)

    # Obstacle detection: is the model planning to go around something?
    hw_now = (np.clip(lines[2, 0] - lines[1, 0], *LANE_WIDTH_CLIP) / 2
              if l_prob * r_prob > 0.5 else self.lane_width.x / 2)
    center_now = (l_prob * (lines[1, 0] + hw_now) + r_prob * (lines[2, 0] - hw_now)) / (l_prob + r_prob)
    plan_y = np.interp(lookahead, model_output['plan'][0, :, 0], model_output['plan'][0, :, 1])
    model_offset = plan_y - np.interp(lookahead, x, center)
    if min(gone) > 0.5:
      self.usual_offset.initialized = False
    usual = self.usual_offset.x if self.usual_offset.initialized else model_offset
    swerve = abs(model_offset - usual) - 0.25 * abs(model_curvature) * lookahead ** 2 / 2
    away = swerve - abs(-center_now - usual)
    lane_curvature = 2.0 * np.polyfit(x, center, 2)[0]
    inside_curve = (abs(lane_curvature) * lookahead ** 2 / 2 > 0.1 and
                    (model_offset - usual) * lane_curvature > 0)
    self.since_turning = (0.0 if abs(model_curvature) * lookahead ** 2 / 2 > 0.5
                          else self.since_turning + DT_MDL)
    if inside_curve:
      self.going_around = False
    elif away > YIELD[0] and self.since_turning > 2.0:
      self.going_around = True
    elif swerve < YIELD[0] / 2:
      self.going_around = False
    self.usual_offset.update_alpha(20.0 if self.going_around else 10.0)
    self.usual_offset.update(model_offset)

    # Reduce trust when the model is likely avoiding an obstacle
    if inside_curve:
      lane_trust *= np.interp(abs(model_offset) - abs(center_now) - 0.25 * abs(model_curvature) * lookahead ** 2 / 2,
                              (0.5, 1.0), [1.0, 0.0])
    else:
      lane_trust *= (np.interp(swerve, (YIELD[0] / 2, YIELD[0]), [1.0, 0.0]) if self.going_around
                     else np.interp(away, YIELD, [1.0, 0.0]))

    if lane_trust < self.trust:
      dk = float(np.clip(dk, min(self.dk, 0.0), max(self.dk, 0.0)))
    self.trust = (lane_trust if lane_trust < self.trust
                  else self.trust + (lane_trust - self.trust) * DT_MDL / (1.5 + DT_MDL))
    self.dk = dk
    return float(self.trust * dk)
