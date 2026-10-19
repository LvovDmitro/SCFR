"""Immutable Hugging Face revisions frozen before SCFR rebuttal inference."""

from __future__ import annotations

import re
from typing import Optional


HF_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

FROZEN_HF_MODEL_REVISIONS = {
    "google/medgemma-1.5-4b-it": (
        "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b"
    ),
    "StanfordAIMI/CheXOne": (
        "0c350e6852ea08f9d9baf3b7595c1a10d4849927"
    ),
    "StanfordAIMI/CheXagent-2-3b": (
        "8f19b53a2eceda4c33b0acec6c81fbc293ad80d0"
    ),
    "nvidia/NV-Reason-CXR-3B": (
        "056bd0383b35226554da9dc5866e095df174ae19"
    ),
    "microsoft/maira-2": "795a2b1cd4a310624b4e3d14b5a23e41fd273deb",
}


class ModelRevisionError(RuntimeError):
    """Raised when immutable model provenance cannot be verified."""


def exact_hf_commit(value: str, *, label: str = "model revision") -> str:
    """Return a normalized full commit SHA or reject mutable/short revisions."""

    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    normalized = value.strip().lower()
    if not HF_COMMIT_RE.fullmatch(normalized):
        raise ValueError(
            f"{label} must be a full 40-character hexadecimal Hugging Face "
            "commit SHA"
        )
    return normalized


def frozen_hf_revision(model_id: str) -> str:
    """Return the pre-output frozen revision for a supported audit model."""

    try:
        return FROZEN_HF_MODEL_REVISIONS[model_id]
    except KeyError as exc:
        raise KeyError(f"No frozen Hugging Face revision for {model_id!r}") from exc


def require_resolved_revision(
    requested_revision: str,
    resolved_revision: Optional[str],
) -> str:
    """Fail closed unless runtime provenance equals the requested full SHA."""

    try:
        requested = exact_hf_commit(
            requested_revision,
            label="requested model revision",
        )
    except ValueError as exc:
        raise ModelRevisionError(str(exc)) from exc
    if not resolved_revision:
        raise ModelRevisionError(
            "Runtime model identity exposes no resolved Hugging Face commit"
        )
    try:
        resolved = exact_hf_commit(
            resolved_revision,
            label="resolved model revision",
        )
    except ValueError as exc:
        raise ModelRevisionError(str(exc)) from exc
    if resolved != requested:
        raise ModelRevisionError(
            f"Resolved model revision mismatch: requested={requested} "
            f"resolved={resolved}"
        )
    return resolved

