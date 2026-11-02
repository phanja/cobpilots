"""
Copyright (c) 2026-, Zeph Leggett, sunnypilot, and starpilot contributors.
Licensed under the MIT License.

Blendful UI visualization:
- Colored lane center line on road that transitions from muted grey to bright blue as nudge grows
- Screen-space curved nudge gauge under the torque bar arc
- Optional debug readout box displaying push in cm with directional nudge arrows
- Full integration with LaneCenterOffset and user preferences
"""

import math
from typing import Any

import numpy as np
from openpilot.starpilot.ui.raylib import rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.shader_polygon import draw_polygon
from openpilot.system.ui.lib.text_measure import measure_text_cached

# Colors
BLUE = rl.Color(0, 200, 255, 255)
IDLE_LINE = rl.Color(150, 160, 168, 255)
IDLE_ALPHA = 0.45

# Push scaling
LOOKAHEAD_TIME = 1.5  # s
MIN_LOOKAHEAD = 10.0  # m
COLOR_FULL = 0.2  # 0.2 m (20 cm) of push for full saturation (sqrt scale)
GAUGE_FULL = 0.2  # 0.2 m of push for full gauge deflection

# Geometry & layout
READOUT_W, READOUT_H = 190, 58
LANE_WIDTH_CLIP = (2.8, 4.5)
MAX_OFFSET = 0.3
MIN_CENTER_TO_LINE = 1.1

# Torque bar gauge geometry
TORQUE_RADIUS = 1200
TORQUE_VIEW_W = 476
TORQUE_CENTER_SHIFT = 8
TORQUE_OFFSET = 22
TORQUE_ANGLE_SPAN = 12.7
GAUGE_GAP = 4
GAUGE_THICKNESS = 6
GAUGE_DOT = 4


def pull_at_lookahead(applied_curvature: float, v_ego: float) -> float:
  """Screen push in meters at lookahead.
  Openpilot positive curvature = turning left = screen left (negative).
  Openpilot negative curvature = turning right = screen right (positive).
  """
  lookahead = max(v_ego * LOOKAHEAD_TIME, MIN_LOOKAHEAD)
  return -applied_curvature * (lookahead ** 2) / 2.0


def nudge_share(pull: float, full: float = COLOR_FULL) -> float:
  """0.0 to 1.0 of how hard lane center pushes, sqrt of push up to full m."""
  return math.sqrt(min(abs(pull) / full, 1.0)) if full > 0 else 0.0


def get_param_bool(params: Params, key: str, default: bool = False) -> bool:
  try:
    val = params.get(key)
    if val is None:
      return default
    if isinstance(val, (bytes, bytearray)):
      return val in (b"1", b"true", b"True")
    return str(val).lower() in ("1", "true")
  except Exception:
    return default


class BlendfulUI:
  """Coordinates blendful road line, torque gauge, and debug readout."""

  def __init__(self):
    self.params = Params()
    self._pull = FirstOrderFilter(0.0, 0.5, DT_MDL)
    self._trust = FirstOrderFilter(0.0, 0.5, DT_MDL)
    self._alpha = FirstOrderFilter(0.0, 0.3, DT_MDL)

    self._enabled = get_param_bool(self.params, "BlendfulUI", default=True) and get_param_bool(self.params, "LaneCentering", default=True)
    self._readout_enabled = get_param_bool(self.params, "BlendfulReadout", default=True)
    self._last_param_check = 0.0

    self._center_points = np.empty((0, 2), dtype=np.float32)
    self._line_color = IDLE_LINE
    self._extent = 0.0
    self._active = False
    self._font = gui_app.font(FontWeight.BOLD)

  def _update_params(self, now: float) -> None:
    if now - self._last_param_check > 1.0:
      self._last_param_check = now
      self._enabled = get_param_bool(self.params, "BlendfulUI", default=True) and get_param_bool(self.params, "LaneCentering", default=True)
      self._readout_enabled = get_param_bool(self.params, "BlendfulReadout", default=True)

  def update(self, v_ego: float, applied_curvature: float, lateral_active: bool,
             now: float) -> None:
    self._update_params(now)
    if not self._enabled:
      self._active = False
      self._pull.update(0.0)
      self._trust.update(0.0)
      self._alpha.update(0.0)
      return

    pull = pull_at_lookahead(applied_curvature, v_ego) if lateral_active else 0.0
    self._pull.update(pull)

    is_engaged = lateral_active and abs(applied_curvature) > 1e-7
    trust_target = 1.0 if is_engaged else (0.5 if lateral_active else 0.0)
    self._trust.update(trust_target)
    self._alpha.update(float(lateral_active))

    self._active = lateral_active and self._alpha.x > 0.05
    nudge = nudge_share(self._pull.x, COLOR_FULL)
    self._extent = math.copysign(nudge_share(self._pull.x, GAUGE_FULL), self._pull.x)

    line_alpha = np.clip(self._trust.x, 0.2, 1.0) * (IDLE_ALPHA + (1.0 - IDLE_ALPHA) * nudge)
    self._line_color = rl.color_alpha(rl.color_lerp(IDLE_LINE, BLUE, nudge), line_alpha * self._alpha.x)

  def compute_center_line_3d(self, lane_lines, lane_line_probs, lane_line_stds,
                             offset_m: float = 0.0) -> np.ndarray:
    """Compute variance-weighted 3D center line from lane lines 1 and 2."""
    if len(lane_lines) < 3 or len(lane_line_probs) < 3:
      return np.empty((0, 3), dtype=np.float32)

    left_raw = lane_lines[1].raw_points
    right_raw = lane_lines[2].raw_points
    has_left = left_raw.shape[0] >= 2
    has_right = right_raw.shape[0] >= 2
    if not has_left and not has_right:
      return np.empty((0, 3), dtype=np.float32)

    l_prob = float(lane_line_probs[1]) if has_left else 0.0
    r_prob = float(lane_line_probs[2]) if has_right else 0.0
    if max(l_prob, r_prob) < 0.25:
      return np.empty((0, 3), dtype=np.float32)

    stds = list(lane_line_stds) if len(lane_line_stds) >= 3 else [0.1, 0.1, 0.1]
    l_std = max(float(stds[1]), 0.01)
    r_std = max(float(stds[2]), 0.01)

    if has_left and has_right and l_prob >= 0.25 and r_prob >= 0.25:
      n_pts = min(left_raw.shape[0], right_raw.shape[0])
      left = left_raw[:n_pts]
      right = right_raw[:n_pts]
      half_width = np.clip(right[:, 1] - left[:, 1], *LANE_WIDTH_CLIP) / 2.0 if l_prob * r_prob > 0.3 else 1.8
      l_var = (l_std ** 2) / max(l_prob, 1e-4)
      r_var = (r_std ** 2) / max(r_prob, 1e-4)
      center = left.copy()
      center[:, 1] = (r_var * (left[:, 1] + half_width) + l_var * (right[:, 1] - half_width)) / (l_var + r_var)
    elif has_left and l_prob >= 0.25:
      center = left_raw.copy()
      half_width = 1.8
      center[:, 1] += half_width
    elif has_right and r_prob >= 0.25:
      center = right_raw.copy()
      half_width = 1.8
      center[:, 1] -= half_width
    else:
      return np.empty((0, 3), dtype=np.float32)

    # Apply LaneCenterOffset
    if offset_m != 0.0 and math.isfinite(offset_m):
      if isinstance(half_width, np.ndarray):
        safe_offset = np.minimum(MAX_OFFSET, np.maximum(0.0, half_width - MIN_CENTER_TO_LINE))
      else:
        safe_offset = min(MAX_OFFSET, max(0.0, half_width - MIN_CENTER_TO_LINE))
      center[:, 1] += np.clip(offset_m, -safe_offset, safe_offset)

    return center

  def draw_road_center(self, rect: rl.Rectangle, polygon: np.ndarray) -> None:
    """Draw the blendful center line polygon on road."""
    if not self._active or polygon.size == 0 or self._alpha.x < 0.05:
      return
    draw_polygon(rect, polygon, self._line_color)

  def draw_torque_gauge(self, rect: rl.Rectangle, fade: float = 1.0, scale: float = 1.0) -> None:
    """Draw the curved lane center nudge gauge under the torque bar arc."""
    if not self._active or fade <= 0.0:
      return

    from openpilot.selfdrive.ui.mici.onroad.torque_bar import arc_bar_pts

    half_span = fade * TORQUE_ANGLE_SPAN * min(1.0, rect.width / (TORQUE_VIEW_W * scale)) / 2.0
    radius = TORQUE_RADIUS * scale
    cx = rect.x + rect.width / 2.0 + TORQUE_CENTER_SHIFT
    cy = rect.y + rect.height + radius - TORQUE_OFFSET * scale
    center = np.array([cx, cy], dtype=np.float32)

    r_mid = radius - (GAUGE_GAP + GAUGE_THICKNESS / 2.0) * scale
    thickness = GAUGE_THICKNESS * scale

    # Faint background track
    track = arc_bar_pts(r_mid, thickness, -90.0 - half_span, -90.0 + half_span, cap_radius=thickness / 2.0)
    draw_polygon(rect, track + center, rl.Color(255, 255, 255, int(255 * 0.15 * fade * self._alpha.x)))

    # Nudge fill arc
    if abs(half_span * self._extent) * np.pi / 180.0 * radius >= 1.0:
      fill = arc_bar_pts(r_mid, thickness, -90.0, -90.0 + half_span * self._extent, cap_radius=thickness / 2.0)
      blue_alpha = np.clip(0.3 + 0.7 * self._trust.x, 0.0, 1.0) * fade * self._alpha.x
      draw_polygon(rect, fill + center, rl.color_alpha(BLUE, blue_alpha))

    # Center zero dot
    dot_alpha = int(255 * 0.9 * fade * self._alpha.x)
    rl.draw_circle(int(center[0]), int(center[1] - r_mid), GAUGE_DOT * scale,
                   rl.Color(182, 182, 182, dot_alpha))

  def draw_readout_box(self, rect: rl.Rectangle, top_offset: float = 10.0, scale: float = 1.0) -> None:
    """Draw the debug readout box with push in cm and directional arrows."""
    if not self._active or not self._readout_enabled or self._alpha.x < 0.05:
      return

    pull = self._pull.x
    w, h = READOUT_W * scale, READOUT_H * scale
    x = rect.x + (rect.width - w) / 2.0
    y = rect.y + top_offset
    cx, cy = x + w / 2.0, y + h / 2.0

    # Background card
    rl.draw_rectangle_rounded(rl.Rectangle(x, y, w, h), 0.5, 12, rl.Color(0, 0, 0, int(170 * self._alpha.x)))

    # Number in cm
    text = f"{abs(pull) * 100:.0f}"
    font_size = int(44 * scale)
    text_w = measure_text_cached(self._font, text, font_size, 0).x
    rl.draw_text_ex(self._font, text, rl.Vector2(cx - text_w / 2.0, cy - font_size / 2.0),
                    font_size, 0, rl.Color(255, 255, 255, int(255 * self._alpha.x)))

    # Directional arrows
    active_alpha = np.clip(abs(pull) / 0.1, 0.0, 1.0) * 255 * self._alpha.x
    active_color = rl.color_alpha(BLUE, active_alpha / 255.0)
    idle_color = rl.Color(255, 255, 255, int(40 * self._alpha.x))

    arrow_w, arrow_h = 16 * scale, 12 * scale
    margin = 18 * scale
    lx = x + margin
    rx = x + w - margin

    # Left arrow (pull < 0 = steer left)
    rl.draw_triangle(rl.Vector2(lx, cy),
                     rl.Vector2(lx + arrow_w, cy + arrow_h),
                     rl.Vector2(lx + arrow_w, cy - arrow_h),
                     active_color if pull < 0 else idle_color)

    # Right arrow (pull > 0 = steer right)
    rl.draw_triangle(rl.Vector2(rx - arrow_w, cy - arrow_h),
                     rl.Vector2(rx - arrow_w, cy + arrow_h),
                     rl.Vector2(rx, cy),
                     active_color if pull > 0 else idle_color)

