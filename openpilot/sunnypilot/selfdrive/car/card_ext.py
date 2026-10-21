"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from opendbc.car import structs
from opendbc.sunnypilot.car.stock_ecu import StockEcuState
from openpilot.common.params import Params
from openpilot.sunnypilot.selfdrive.car.stock_ecu_handback import StockEcuHandBackServer


class CardExt:
  """sunnypilot's per-frame hooks into card, one object so card.py carries one-line call sites.

  Ordering contract with card.state_update: update_v_cruise_post runs after update_v_cruise
  and initialize_v_cruise and before CS.vCruise is read, so the reconciled setpoint is the
  one published. sm is card's SubMaster, already updated this frame.
  """

  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParamsSP, params: Params, sm, v_cruise_helper, CI) -> None:
    self.sm = sm
    self.v_cruise_helper = v_cruise_helper
    # The stock ECU transition contract (opendbc/sunnypilot/car/stock_ecu.py): a controller that
    # silences a stock ECU under openpilot longitudinal exposes `stock_ecu_state`; nothing
    # brand-specific is read here. The hand-back server answers the lifecycle's requests off it.
    self.controller = CI.CC
    self.car_state = CI.CS
    self.handback = StockEcuHandBackServer(params)
    self.params = params
    self._last_pcm_dist = None
    self._last_sa_toggle = None

  def update_v_cruise_post(self, CS, CS_SP) -> None:
    helper = self.v_cruise_helper
    # the regime the reconciler gates on, from the same messages this frame saw
    helper.update_plan_regime(self.sm['longitudinalPlanSP'], self.sm['carControlSP'])
    helper.reconcile_setpoint_with_dash(CS)
    # publish the arbiter's session (plannerd mirrors it; the ICBM servo freezes on a prompt)
    helper.cruise_arbiter.fill_msg(CS_SP)
    # whether ICBM drives the set speed: card's call, read by every other process
    CS_SP.zoompilot.icbmActivation = helper.icbm_latch.activation
    # the driver's view of the stock ECU, for the engage-press alert (stockEcuNotReady)
    CS_SP.zoompilot.stockEcu = str(self.stock_ecu_state)
    # a second, opposite-direction distance button, on the brands whose carstate exposes one
    CS_SP.zoompilot.distanceFarther = bool(getattr(self.car_state, "distance_more_button", 0))
    # the EPS re-arming after the car's lane keep came back on, on the brands that track it
    CS_SP.zoompilot.lkasArming = bool(getattr(self.car_state, "lkas_arming", False))

    # Sync Toyota dashboard follow distance to openpilot personality:
    # 1: Far (3 bars) -> 2: Relaxed, 2: Med (2 bars) -> 1: Standard, 3: Close (1 bar) -> 0: Aggressive
    pcm_dist = getattr(self.car_state, "pcm_follow_distance", 0)
    if pcm_dist in (1, 2, 3) and pcm_dist != self._last_pcm_dist:
      self._last_pcm_dist = pcm_dist
      self.params.put('LongitudinalPersonality', 3 - int(pcm_dist))

    # Toggle Jetlink Big Model paused state on Toyota LKA button hold (LDA_SA_TOGGLE transition):
    lkas_hud = getattr(self.car_state, "lkas_hud", {})
    sa_toggle = lkas_hud.get("LDA_SA_TOGGLE", 0)
    if self._last_sa_toggle is not None and sa_toggle in (1, 2) and sa_toggle != self._last_sa_toggle:
      paused = not self.params.get_bool("JetlinkPaused")
      self.params.put_bool("JetlinkPaused", paused)
    if sa_toggle in (1, 2):
      self._last_sa_toggle = sa_toggle

  @property
  def stock_ecu_state(self) -> StockEcuState:
    return getattr(self.controller, "stock_ecu_state", StockEcuState.NOT_NEEDED)

  def controls_update(self, CS, CC, CC_SP: structs.CarControlSP) -> structs.CarControlSP:
    """Runs just before CI.apply on the converted CarControlSP struct, which it may edit."""
    self.v_cruise_helper.cruise_arbiter.gate_send_button(CC_SP)
    self.handback.update(CC.enabled, self.stock_ecu_state, CC_SP)
    return CC_SP

  def update_params(self) -> None:
    # rides card's params thread, keeping param reads off the 100 Hz path
    self.handback.update_params()
