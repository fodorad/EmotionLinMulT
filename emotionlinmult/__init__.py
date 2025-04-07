from importlib.metadata import version

try:
    __version__ = version("exordium")
except Exception:
    __version__ = "unknown"