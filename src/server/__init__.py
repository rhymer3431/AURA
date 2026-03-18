"""Server package.

Keep package initialization lightweight so read-only consumers such as the
dashboard backend can import `server.snapshot_adapter` without pulling in the
full runtime stack.
"""

__all__: list[str] = []
