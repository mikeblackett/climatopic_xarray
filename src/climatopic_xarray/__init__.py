from .bounds import CFBoundsDatasetAccessor

__version__ = 'unknown'
__author__ = 'Mike Blackett'

try:
    from _version import __version__  # noqa: F401
except ImportError:
    pass


__all__ = [
    'CFBoundsDatasetAccessor',
    '__version__',
]
