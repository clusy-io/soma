"""A research-grade switch controller: transactional and crash-recoverable.

Drives the platform API for sandbox lifecycle and execution, and owns the
protocol: validate the destination BEFORE releasing the source, persist every
phase to a journal so a dead controller can be restarted and resume, make
retries idempotent, and fence the source after commit.
"""
