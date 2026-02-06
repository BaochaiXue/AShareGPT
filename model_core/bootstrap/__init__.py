"""Composition root utilities."""

from .container import LegacyContainer, create_legacy_container
from .factories import (
    create_training_engine,
    create_training_workflow_service,
    create_training_workflow_service_from_components,
)

__all__ = [
    "LegacyContainer",
    "create_legacy_container",
    "create_training_engine",
    "create_training_workflow_service",
    "create_training_workflow_service_from_components",
]
