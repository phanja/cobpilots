import sys
import json
import ctypes
import weakref
import builtins
import datetime
from pathlib import Path
from enum import IntEnum, IntFlag

from openpilot.common.swaglog import cloudlog
from openpilot.common.params_paths import params_path


class ParamKeyFlag(IntFlag):
  PERSISTENT = 0x02
  CLEAR_ON_MANAGER_START = 0x04
  CLEAR_ON_ONROAD_TRANSITION = 0x08
  CLEAR_ON_OFFROAD_TRANSITION = 0x10
  DEVELOPMENT_ONLY = 0x40
  CLEAR_ON_IGNITION_ON = 0x80
  ALL = 0xFFFFFFFF


class ParamKeyType(IntEnum):
  STRING = 0
  BOOL = 1
  INT = 2
  FLOAT = 3
  TIME = 4
  JSON = 5
  BYTES = 6


_suffix = ".dylib" if sys.platform == "darwin" else ".so"
lib = ctypes.CDLL(Path(__file__).with_name(f"libparams_c{_suffix}"))

ParamsHandle = ctypes.c_void_p


class ParamsBuffer(ctypes.Structure):
  _fields_ = [("data", ctypes.c_void_p), ("size", ctypes.c_size_t)]


def _bind_raw(name, args, result=None):
  function = getattr(lib, name)
  function.argtypes = args
  function.restype = result
  return function


params_last_error = _bind_raw("params_last_error", [], ctypes.c_char_p)


def _bind(name, args, result=None):
  function = _bind_raw(name, args, result)

  def checked(*call_args):
    value = function(*call_args)
    if error := params_last_error():
      raise RuntimeError(error.decode())
    return value

  return checked


params_create = _bind("params_create", [ctypes.c_char_p, ctypes.c_size_t], ParamsHandle)
params_destroy = _bind("params_destroy", [ParamsHandle])
params_clear_all = _bind("params_clear_all", [ParamsHandle, ctypes.c_uint])
params_check_key = _bind("params_check_key", [ParamsHandle, ctypes.c_char_p], ctypes.c_bool)
params_get_key_type = _bind("params_get_key_type", [ParamsHandle, ctypes.c_char_p], ctypes.c_int)
params_get_default = _bind("params_get_default", [ParamsHandle, ctypes.c_char_p], ParamsBuffer)
params_get = _bind("params_get", [ParamsHandle, ctypes.c_char_p, ctypes.c_bool], ParamsBuffer)
params_get_bool = _bind("params_get_bool", [ParamsHandle, ctypes.c_char_p, ctypes.c_bool], ctypes.c_bool)
params_put = _bind("params_put", [ParamsHandle, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_bool], ctypes.c_int)
params_put_bool = _bind("params_put_bool", [ParamsHandle, ctypes.c_char_p, ctypes.c_bool, ctypes.c_bool], ctypes.c_int)
params_remove = _bind("params_remove", [ParamsHandle, ctypes.c_char_p], ctypes.c_int)
params_get_path = _bind("params_get_path", [ParamsHandle, ctypes.c_char_p, ctypes.c_size_t], ParamsBuffer)
params_keys_size = _bind("params_keys_size", [ParamsHandle], ctypes.c_size_t)
params_key_at = _bind("params_key_at", [ParamsHandle, ctypes.c_size_t], ParamsBuffer)

PYTHON_2_CPP = {
  (str, ParamKeyType.STRING): lambda v: v,
  (builtins.bool, ParamKeyType.BOOL): lambda v: "1" if v else "0",
  (int, ParamKeyType.INT): str,
  (float, ParamKeyType.FLOAT): str,
  (datetime.datetime, ParamKeyType.TIME): lambda v: v.isoformat(),
  (dict, ParamKeyType.JSON): json.dumps,
  (list, ParamKeyType.JSON): json.dumps,
  (bytes, ParamKeyType.BYTES): lambda v: v,
}
CPP_2_PYTHON = {
  ParamKeyType.STRING: lambda v: v.decode("utf-8"),
  ParamKeyType.BOOL: lambda v: v == b"1",
  ParamKeyType.INT: int,
  ParamKeyType.FLOAT: float,
  ParamKeyType.TIME: lambda v: datetime.datetime.fromisoformat(v.decode("utf-8")),
  ParamKeyType.JSON: json.loads,
  ParamKeyType.BYTES: lambda v: v,
}


def ensure_bytes(v):
  return v.encode() if isinstance(v, str) else v


def _copy_string(value):
  if value.data is None:
    return None
  return ctypes.string_at(value.data, value.size)


class UnknownKeyName(Exception):
  pass


def _load_keys_header():
  keys = {}
  defaults = {}
  try:
    header = Path(__file__).parent / "params_keys.h"
    with open(header, "r") as f:
      for line in f:
        line = line.strip()
        if line.startswith('{"') and '",' in line:
          name = line.split('{"', 1)[1].split('",', 1)[0]
          type_str = "STRING"
          for t in ("BOOL", "INT", "FLOAT", "TIME", "JSON", "BYTES", "STRING"):
            if f", {t}" in line or f" {t}," in line or f" {t}}}" in line:
              type_str = t
              break
          k_bytes = name.encode()
          keys[k_bytes] = getattr(ParamKeyType, type_str, ParamKeyType.STRING)
          parts = [p.strip() for p in line.split(",")]
          for p in parts[2:]:
            p_clean = p.rstrip("}").rstrip(";").strip()
            if p_clean.startswith('"') and p_clean.endswith('"'):
              defaults[k_bytes] = p_clean.strip('"').encode()
  except Exception:
    pass
  return keys, defaults


HEADER_KEYS, HEADER_DEFAULTS = _load_keys_header()


class Params:
  def __init__(self, d=""):
    path = ensure_bytes(params_path(d))
    self.p = params_create(path, len(path))
    self._finalizer = weakref.finalize(self, params_destroy, self.p)
    self._finalizer.atexit = False  # daemon threads can still be using live Params handles during interpreter shutdown
    self.d = d

  def __reduce__(self):
    return (type(self), (self.d,))

  def clear_all(self, tx_flag=ParamKeyFlag.ALL):
    params_clear_all(self.p, int(tx_flag))

  def check_key(self, key):
    key = ensure_bytes(key)
    if b"\0" in key:
      raise UnknownKeyName(key)
    if params_check_key(self.p, key) or key in HEADER_KEYS:
      return key
    raise UnknownKeyName(key)

  def python2cpp(self, proposed_type, expected_type, value, key):
    cast = PYTHON_2_CPP.get((proposed_type, expected_type))
    if cast:
      return cast(value)
    raise TypeError(f"Type mismatch while writing param {key}: {proposed_type=} {expected_type=} {value=}")

  def _cpp2python(self, t, value, default, key):
    if value is None:
      return None
    try:
      return CPP_2_PYTHON[t](value)
    except (KeyError, TypeError, ValueError):
      cloudlog.warning(f"Failed to cast param {key} with {value=} from type {t=}")
      return self._cpp2python(t, default, None, key)

  def _default(self, key):
    k = self.check_key(key)
    if k in HEADER_KEYS and not params_check_key(self.p, k):
      return HEADER_DEFAULTS.get(k)
    return _copy_string(params_get_default(self.p, k))

  def get(self, key, block=False, return_default=False):
    k = self.check_key(key)
    t = self.get_type(k)
    default = self._default(k) if return_default else None
    value = _copy_string(params_get(self.p, k, block))
    if value == b"":
      if block:
        raise KeyboardInterrupt
      return self._cpp2python(t, default, None, key)
    return self._cpp2python(t, value, default, key)

  def get_bool(self, key, block=False, return_default=False):
    k = self.check_key(key)
    raw = _copy_string(params_get(self.p, k, block))
    if raw == b"":
      if return_default:
        default = self._default(k)
        if default is not None:
          return default in (b"1", "1", 1, True, b"true", b"True")
      return False
    return raw in (b"1", "1", 1, True, b"true", b"True")

  def _put_cast(self, key, dat):
    return ensure_bytes(self.python2cpp(type(dat), self.get_type(key), dat, key))

  def put(self, key, dat, block=False):
    """Write a parameter. block=True waits until it is persisted to disk."""
    k = self.check_key(key)
    value = self._put_cast(k, dat)
    params_put(self.p, k, value, len(value), block)

  def put_bool(self, key, val, block=False):
    params_put_bool(self.p, self.check_key(key), val, block)

  def remove(self, key):
    params_remove(self.p, self.check_key(key))

  def get_param_path(self, key=""):
    key = ensure_bytes(key)
    return _copy_string(params_get_path(self.p, key, len(key))).decode()

  def get_type(self, key):
    k = self.check_key(key)
    if k in HEADER_KEYS and not params_check_key(self.p, k):
      return HEADER_KEYS[k]
    return ParamKeyType(params_get_key_type(self.p, k))

  def all_keys(self):
    keys = []
    for i in range(params_keys_size(self.p)):
      keys.append(_copy_string(params_key_at(self.p, i)))
    for k in HEADER_KEYS:
      if k not in keys:
        keys.append(k)
    return keys

  def get_default_value(self, key):
    k = self.check_key(key)
    return self._cpp2python(self.get_type(k), self._default(k), None, key)

  def cpp2python(self, key, value):
    return self._cpp2python(self.get_type(key), value, None, key)


if __name__ == "__main__":
  import sys

  params = Params()
  key = sys.argv[1]
  params.check_key(key)
  if len(sys.argv) == 3:
    val = sys.argv[2]
    print(f"SET: {key} = {val}")
    params.put(key, val, block=True)
  elif len(sys.argv) == 2:
    print(f"GET: {key} = {params.get(key)}")
