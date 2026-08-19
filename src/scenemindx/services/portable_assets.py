"""Project-relative asset locators for portable local inference indexes."""

from __future__ import annotations

from pathlib import Path


PORTABLE_ASSET_SCHEME = "scenemindx_portable_asset_uri_v1"


class PortableAssetResolver:
    """Resolve and serialize bounded project/course asset locators.

    Portable manifests never persist a drive letter or a server absolute path.
    Historical absolute paths remain readable for backward compatibility, but
    only the three explicit URI schemes are written to the standard local
    index.
    """

    def __init__(
        self,
        *,
        project_root: Path,
        train_root: Path,
        val_root: Path,
    ) -> None:
        self.project_root = project_root.resolve()
        self.train_root = train_root.resolve()
        self.val_root = val_root.resolve()

    @staticmethod
    def _bounded(root: Path, relative: str) -> Path:
        normalized = relative.replace("\\", "/").lstrip("/")
        candidate = (root / normalized).resolve()
        if candidate != root and not candidate.is_relative_to(root):
            raise ValueError("portable_asset_path_traversal")
        return candidate

    def resolve(self, locator: str | Path) -> Path:
        """Resolve the requested value."""
        value = str(locator).strip()
        if value.startswith("project://"):
            return self._bounded(
                self.project_root,
                value.removeprefix("project://"),
            )
        if value.startswith("course-train://"):
            return self._bounded(
                self.train_root,
                value.removeprefix("course-train://"),
            )
        if value.startswith("course-val://"):
            return self._bounded(
                self.val_root,
                value.removeprefix("course-val://"),
            )
        if "://" in value:
            raise ValueError("unsupported_portable_asset_scheme")
        return Path(value).resolve()

    def serialize(self, locator: str | Path) -> str:
        """Execute the serialize operation."""
        path = self.resolve(locator)
        roots = (
            ("project", self.project_root),
            ("course-train", self.train_root),
            ("course-val", self.val_root),
        )
        for scheme, root in roots:
            if path == root or path.is_relative_to(root):
                relative = path.relative_to(root).as_posix()
                return f"{scheme}://{relative}"
        raise ValueError("asset_outside_portable_roots")

