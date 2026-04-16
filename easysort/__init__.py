"""Lazy public exports for the easysort package."""
from __future__ import annotations

from importlib import import_module

__all__ = [
  "RegistryBase",
  "RegistryConnector",
  "Runner",
  "RunnerJob",
  "PusherJob",
  "ContinuousRunner",
  "Concat",
  "REGISTRY_LOCAL_IP",
]


def __getattr__(name: str):
  module_name = None
  if name in {"RegistryBase", "RegistryConnector"}:
    module_name = "easysort.registry"
  elif name in {"Runner", "RunnerJob", "PusherJob", "ContinuousRunner"}:
    module_name = "easysort.runner"
  elif name in {"Concat", "REGISTRY_LOCAL_IP"}:
    module_name = "easysort.helpers"

  if module_name is None:
    raise AttributeError(f"module 'easysort' has no attribute {name!r}")

  module = import_module(module_name)
  value = getattr(module, name)
  globals()[name] = value
  return value
