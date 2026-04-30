"""Typed exceptions for napari-deeplabcut core utilities.

They are used by IO routing/provenance helpers to express deterministic
failure modes (e.g., ambiguity) without relying on ad-hoc strings.

No behavior changes are introduced by merely defining these exceptions.
"""

# src/napari_deeplabcut/core/errors.py
from __future__ import annotations


class NapariDLCError(RuntimeError):
    """Base class for napari-deeplabcut domain errors."""


class MissingProvenanceError(NapariDLCError):
    """Raised when required provenance is absent (cannot determine save target)."""


class AmbiguousSaveError(NapariDLCError):
    """Raised when multiple save targets are plausible and policy forbids guessing."""

    def __init__(self, message: str, candidates: list[str] | None = None):
        super().__init__(message)
        self.candidates = candidates or []


class UnresolvablePathError(NapariDLCError):
    """Raised when provenance exists but cannot be resolved to a concrete filesystem path."""
