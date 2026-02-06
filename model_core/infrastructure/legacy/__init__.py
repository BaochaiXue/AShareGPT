"""Legacy adapters kept for migration compatibility."""

from warnings import warn

from model_core.infrastructure.adapters.trainer import LegacyAlphaTrainer

DEPRECATED_SINCE = "v0.5.0"
REMOVAL_VERSION = "v0.6.0"

warn(
    (
        "model_core.infrastructure.legacy is deprecated since "
        f"{DEPRECATED_SINCE} and will be removed in {REMOVAL_VERSION}. "
        "Import from model_core.infrastructure.adapters instead."
    ),
    FutureWarning,
    stacklevel=2,
)

__all__ = ["LegacyAlphaTrainer", "DEPRECATED_SINCE", "REMOVAL_VERSION"]
