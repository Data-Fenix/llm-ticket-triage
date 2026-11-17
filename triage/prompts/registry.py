"""Versioned prompt registry.

Prompts are markdown files with a frontmatter header. They are data, not code,
so a prompt change is a reviewable diff, is pinned by version in every output
record, and can be evaluated against the previous version before rollout.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from ..config import settings

FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)


@dataclass(frozen=True)
class Prompt:
    version: str
    system: str
    notes: str
    sha: str  # content hash so a silently edited file cannot masquerade as an old version

    @property
    def label(self) -> str:
        return f"{self.version}@{self.sha}"


def _parse(path: Path) -> Prompt:
    raw = path.read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    m = FRONTMATTER.match(raw)
    body = raw
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        body = raw[m.end():]
    system = body.strip()
    return Prompt(
        version=meta.get("version", path.stem),
        system=system,
        notes=meta.get("notes", ""),
        sha=hashlib.sha256(system.encode("utf-8")).hexdigest()[:8],
    )


def list_versions(prompts_dir: Path | None = None) -> list[str]:
    d = prompts_dir or settings.prompts_dir
    return sorted(p.stem for p in d.glob("v*.md"))


def load_prompt(version: str | None = None, prompts_dir: Path | None = None) -> Prompt:
    d = prompts_dir or settings.prompts_dir
    version = version or settings.prompt_version
    path = d / f"{version}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt {version!r} not found. Available: {list_versions(d)}")
    return _parse(path)
