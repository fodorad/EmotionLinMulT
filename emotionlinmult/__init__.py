from importlib.metadata import version

try:
    __version__ = version("emotionlinmult")
except Exception:
    __version__ = "unknown"
