"""Capsule size guard for the fixture's optional padding.

`fixture.build_transported_namespace` can pad the namespace with an incompressible array
(`FixtureSpec.padding_bytes`) to exercise large capsules. The platform the
cloud experiments ran on refuses any checkpoint larger than 4 GiB (inclusive:
only strictly greater sizes are refused), so a padding request past that is a
harness error and is rejected here rather than silently truncated.

None of the shipped harnesses sets `padding_bytes`; the guard is kept so the
fixture's import resolves and the limit stays explicit.
"""

from __future__ import annotations

MiB = 1024 ** 2
GiB = 1024 ** 3

#: Largest capsule the platform accepts, in bytes (inclusive).
CAPSULE_MAX_BYTES = 4 * GiB


class CapsuleTooLarge(ValueError):
    """Raised when a requested size would be refused by the platform."""


def validate_size(nbytes: int, *, label: str = "", ceiling: int = CAPSULE_MAX_BYTES) -> int:
    """Return ``nbytes``, or raise if a capsule of this size would be refused."""
    if nbytes > ceiling:
        raise CapsuleTooLarge(
            f"{label or nbytes} is {nbytes - ceiling:,} bytes over the "
            f"{ceiling:,}-byte capsule ceiling")
    return nbytes
