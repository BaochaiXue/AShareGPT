"""Composition root utilities."""

from .container import LegacyContainer, create_legacy_container
from .factories import create_training_engine

__all__ = ["LegacyContainer", "create_legacy_container", "create_training_engine"]
