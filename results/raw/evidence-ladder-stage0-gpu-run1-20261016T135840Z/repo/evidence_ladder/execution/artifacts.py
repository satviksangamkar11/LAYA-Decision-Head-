"""Content-addressed immutable artifact store. A candidate's source is written once under its sha256 (mode 0444); the hash is re-verified from the stored bytes immediately
before every execution, and the bytes that were hashed are the bytes that are executed (callers get the bytes back, never a path), so there is no check/use gap.
The store directory is never mounted into a jail, so a candidate cannot reach it."""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Optional

from evidence_ladder.common.interfaces import Artifact, sha256_text


class ArtifactIntegrityError(Exception):
    """Stored bytes or the in-memory artifact do not match the content address. Callers report status 'error' / 'sandbox_error', never a candidate failure."""


class ArtifactStore:
    def __init__(self, root: Optional[os.PathLike] = None):
        self.root = Path(root) if root else Path(tempfile.mkdtemp(prefix="el-artifacts-"))
        self.root.mkdir(parents=True, exist_ok=True)

    def path_of(self, artifact_id: str) -> Path:
        return self.root / artifact_id[:2] / (artifact_id + ".src")

    def put(self, artifact: Artifact) -> Path:
        if sha256_text(artifact.source) != artifact.artifact_id:
            raise ArtifactIntegrityError(f"artifact {artifact.artifact_id[:12]} does not match its source hash")
        p = self.path_of(artifact.artifact_id)
        if p.exists():          # write-once: never rewrite; verify() below detects a changed file
            return p
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".tmp-")
        with os.fdopen(fd, "wb") as f:
            f.write(artifact.source.encode("utf-8"))
        os.chmod(tmp, 0o444)
        os.replace(tmp, p)
        return p

    def verified_bytes(self, artifact: Artifact) -> bytes:
        """Store (if new), then re-read and re-hash. Raises ArtifactIntegrityError on any mismatch."""
        p = self.put(artifact)
        try:
            raw = p.read_bytes()
        except OSError as e:
            raise ArtifactIntegrityError(f"cannot read stored artifact: {e}") from e
        sha = hashlib.sha256(raw).hexdigest()
        if sha != artifact.artifact_id:
            raise ArtifactIntegrityError(f"stored bytes hash {sha[:12]} != artifact id {artifact.artifact_id[:12]}")
        return raw
