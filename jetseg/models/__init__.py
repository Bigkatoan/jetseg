"""Model utilities for jetseg.

Expose available model classes for easy import:

from jetseg.models import UNetOptimized, count_parameters

This package contains small, optimized model variants intended for
edge deployment and fast iteration.
"""
from .optimized_unet import UNetOptimized, count_parameters

__all__ = ["UNetOptimized", "count_parameters"]
