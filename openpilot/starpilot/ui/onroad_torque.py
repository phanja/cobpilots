"""Shared StarPilot torque bar with caller-qualified utilization."""

import time
from openpilot.starpilot.ui.raylib import rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.starpilot.ui.blendful_renderer import BlendfulUI
from openpilot.starpilot.ui.onroad_state import OnroadState
from openpilot.starpilot.ui.onroad_customization import offset as widget_offset, placement
from openpilot.starpilot.ui.onroad_torque_draw import TorqueBarPainter, arc_scale
from openpilot.starpilot.ui.onroad_torque_geometry import CAP_RADIUS, LARGE_CAP_RADIUS


class TorqueBarWidget:
  def __init__(self):
    self._torque_filter = FirstOrderFilter(0.0, 0.1, 1 / 60)
    self._alpha_filter = FirstOrderFilter(0.0, 0.1, 1 / 60)
    self._blendful_ui = BlendfulUI()
    self._drive_frame = None
    self._painter = TorqueBarPainter()

  def prepare(self) -> None:
    self._painter.prepare()

  def render(self, rect: rl.Rectangle, state: OnroadState, screen_width: float) -> None:
    drive_frame = getattr(state, 'torque_drive_frame', None)
    if drive_frame != self._drive_frame:
      self._torque_filter.x = self._alpha_filter.x = 0.0
      self._drive_frame = drive_frame
    # Projected canvases use Large UI placements even when their width is 1860.
    profile = 'large' if screen_width >= 1860 else 'compact'
    enabled = placement(state.customization, profile, 'torque_bar')['enabled']
    if not state.lateral_active or not getattr(state, 'torque_source_available', True):
      self._torque_filter.x = 0.0
      self._alpha_filter.x = 0.0
      return
    utilization = max(-1.0, min(1.0, state.torque_utilization))
    torque = self._torque_filter.update(utilization)
    alpha = self._alpha_filter.update(1.0)
    if not enabled:
      return
    cap_radius = LARGE_CAP_RADIUS if screen_width == 2160 else CAP_RADIUS
    self._painter.render(rect, torque, alpha, screen_width, cap_radius, widget_offset(state.customization, profile, 'torque_bar'))

    applied = 0.0
    lateral_active = bool(state.lateral_active)
    if 'starpilotLateralState' in ui_state.sm.seen and ui_state.sm.alive['starpilotLateralState']:
      lc = ui_state.sm['starpilotLateralState'].laneCentering
      applied = float(lc.appliedCorrection)
      lateral_active = bool(lc.lateralActive and lateral_active)
    v_ego = float(ui_state.sm['carState'].vEgo) if ui_state.sm.seen['carState'] else 0.0
    self._blendful_ui.update(v_ego, applied, lateral_active, time.monotonic())
    scale = arc_scale(rect.width, rect.height, screen_width)
    self._blendful_ui.draw_torque_gauge(rect, fade=alpha, scale=scale)

  def close(self) -> None:
    self._painter.close()
