"""Bounded Hugging Face resolution used by the local control plane.

The loading contracts deliberately do not contact the Hub.  The browser/server
workflow needs one small, auditable bridge from a user supplied model revision
(``main``, a tag, or a commit) to the immutable commit required by
``LoadingPlan``.  This module owns that bridge and nothing else: it never loads
weights, imports a model runtime, or accepts credentials from request data.
"""

from __future__ import annotations

import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from ..loading import (
    DeviceKind,
    DownloadPolicy,
    DTypePolicy,
    HuggingFaceSource,
    ImmutableRevisionEvidence,
    LoadConfig,
    LoadingPlan,
    ResolvedSource,
    RevisionEvidenceKind,
    TokenizerRequest,
)

HUB_RESOLUTION_SCHEMA_VERSION = "1.0"
HUGGINGFACE_API_ORIGIN = "https://huggingface.co/api"

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_MAX_RESPONSE_BYTES = 512 * 1024
_DEFAULT_TIMEOUT_SECONDS = 20.0


class ModelResolutionError(RuntimeError):
    """A safe, fixed-stage failure while resolving public Hub metadata."""

    def __init__(
        self,
        stage: str,
        message: str | None = None,
        *,
        cause: BaseException | None = None,
    ) -> None:
        if stage not in {"request", "response", "identity", "dependency"}:
            raise ValueError("unsupported model-resolution stage")
        self.stage = stage
        text = f"Hugging Face resolution failed at {stage}"
        if message:
            text = f"{text}: {message}"
        super().__init__(text)
        if cause is not None:
            self.__cause__ = cause


def _validated_identifier(value: object, field_name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ModelResolutionError("identity", f"{field_name} is invalid")
    if len(value) > 500 or any(ord(char) < 32 for char in value):
        raise ModelResolutionError("identity", f"{field_name} is invalid")
    # HF repository IDs are deliberately treated as a single path segment per
    # owner/name.  quote() below prevents user text from widening the endpoint.
    if "/" not in value or value.startswith("/") or value.endswith("/"):
        raise ModelResolutionError("identity", f"{field_name} must use owner/name form")
    return value


def _validated_revision(value: object, field_name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ModelResolutionError("identity", f"{field_name} is invalid")
    if len(value) > 200 or any(ord(char) < 32 for char in value):
        raise ModelResolutionError("identity", f"{field_name} is invalid")
    return value


def _api_url(kind: str, identifier: str, revision: str) -> str:
    # quote the repository as two escaped path components, not one opaque URL.
    owner, name = identifier.split("/", 1)
    encoded = f"{quote(owner, safe='')}/{quote(name, safe='')}"
    return f"{HUGGINGFACE_API_ORIGIN}/{kind}/{encoded}?revision={quote(revision, safe='')}"


def _fetch_revision(
    kind: str,
    identifier: str,
    revision: str,
    *,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> tuple[str, str]:
    if kind not in {"models", "datasets"}:
        raise ValueError("kind must be models or datasets")
    url = _api_url(kind, identifier, revision)
    headers = {"Accept": "application/json", "User-Agent": "moeatlas-local/1.0"}
    # Private/gated repositories can use the process credential configured for
    # the runtime; credentials never enter the browser DTO or an error body.
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
    if token and len(token) <= 512 and not any(ord(char) < 32 for char in token):
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS origin
            body = response.read(_MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        # Do not echo the repository or Hub body into the API response.
        raise ModelResolutionError(
            "request", "the public Hub revision could not be resolved", cause=exc
        ) from exc
    except (OSError, URLError, TimeoutError) as exc:
        raise ModelResolutionError(
            "request", "the public Hub is unavailable", cause=exc
        ) from exc
    if len(body) > _MAX_RESPONSE_BYTES:
        raise ModelResolutionError(
            "response", "the public Hub response exceeded the metadata budget"
        )
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelResolutionError(
            "response", "the public Hub returned invalid metadata", cause=exc
        ) from exc
    if not isinstance(payload, dict):
        raise ModelResolutionError(
            "response", "the public Hub returned an unexpected metadata shape"
        )
    resolved = payload.get("sha")
    if type(resolved) is not str or _COMMIT.fullmatch(resolved) is None:
        raise ModelResolutionError(
            "response", "the public Hub did not return an immutable commit"
        )
    return resolved, url


def resolve_huggingface_revision(
    identifier: str,
    requested_revision: str = "main",
    *,
    kind: str = "models",
) -> tuple[str, str]:
    """Resolve one model/dataset revision to a lowercase 40-character commit."""

    stable_identifier = _validated_identifier(identifier, "repository identifier")
    stable_revision = _validated_revision(requested_revision, "requested revision")
    if _COMMIT.fullmatch(stable_revision):
        return stable_revision, "caller supplied immutable commit"
    return _fetch_revision(kind, stable_identifier, stable_revision)


def resolve_huggingface_plan(
    model_id: str,
    requested_revision: str = "main",
    *,
    device: str = DeviceKind.AUTO.value,
    dtype: str = DTypePolicy.PRESERVE.value,
    trust_remote_code: bool = False,
    allow_downloads: bool = True,
) -> LoadingPlan:
    """Build a fully resolved, download-enabled Hugging Face ``LoadingPlan``."""

    stable_model_id = _validated_identifier(model_id, "model identifier")
    stable_requested = _validated_revision(requested_revision, "model revision")
    if type(allow_downloads) is not bool or type(trust_remote_code) is not bool:
        raise ModelResolutionError("identity", "the loading policy is invalid")
    try:
        dtype_policy = DTypePolicy(dtype)
        device_kind = device
        if type(device_kind) is not str:
            raise ValueError("device must be a string")
        source = HuggingFaceSource(
            model_id=stable_model_id,
            requested_revision=stable_requested,
            tokenizer=TokenizerRequest(
                identifier=stable_model_id, inherit_model_revision=True
            ),
            download_policy=DownloadPolicy.ALLOW_DOWNLOADS
            if allow_downloads
            else DownloadPolicy.OFFLINE,
            allow_downloads=allow_downloads,
        )
        config = LoadConfig(
            device=device_kind,
            dtype=dtype_policy,
            trust_remote_code=trust_remote_code,
            remote_code_acknowledged=trust_remote_code,
            download_policy=source.download_policy,
            allow_downloads=allow_downloads,
        )
    except (TypeError, ValueError) as exc:
        raise ModelResolutionError("identity", "the loading policy is invalid", cause=exc) from exc

    if not allow_downloads and _COMMIT.fullmatch(stable_requested) is None:
        raise ModelResolutionError(
            "dependency",
            "offline loading requires an immutable model commit revision",
        )

    resolved_revision, evidence_source = resolve_huggingface_revision(
        stable_model_id, stable_requested, kind="models"
    )
    evidence = ImmutableRevisionEvidence(
        kind=RevisionEvidenceKind.GIT_COMMIT,
        digest=resolved_revision,
        evidence_source=evidence_source,
    )
    resolution = ResolvedSource(
        source_type=source.source_type,
        model_id=source.model_id,
        requested_model_revision=source.requested_revision,
        resolved_model_revision=resolved_revision,
        resolved_model_revision_evidence=evidence,
        requested_tokenizer_revision=source.requested_revision,
        resolved_tokenizer_revision=resolved_revision,
        resolved_tokenizer_revision_evidence=evidence,
        resolution_method="huggingface_public_api",
    )
    return LoadingPlan(source=source, config=config, resolution=resolution)


def resolve_huggingface_dataset_revision(
    dataset_id: str,
    requested_revision: str = "main",
    *,
    allow_downloads: bool = True,
) -> str:
    """Return an immutable dataset commit for provenance and deterministic reads."""

    if type(allow_downloads) is not bool:
        raise TypeError("allow_downloads must be an exact bool")
    stable_requested = _validated_revision(requested_revision, "requested dataset revision")
    if not allow_downloads and _COMMIT.fullmatch(stable_requested) is None:
        raise ModelResolutionError(
            "dependency",
            "offline loading requires an immutable dataset commit revision",
        )
    resolved, _ = resolve_huggingface_revision(dataset_id, stable_requested, kind="datasets")
    return resolved


__all__ = [
    "HUB_RESOLUTION_SCHEMA_VERSION",
    "HUGGINGFACE_API_ORIGIN",
    "ModelResolutionError",
    "resolve_huggingface_dataset_revision",
    "resolve_huggingface_plan",
    "resolve_huggingface_revision",
]
