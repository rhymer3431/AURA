from __future__ import annotations

from collections import deque
from pathlib import Path


class LogTailer:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir)

    def get_recent(self, *, limit: int = 200) -> list[dict[str, object]]:
        if not self.log_dir.exists():
            return []
        records: list[dict[str, object]] = []
        files = sorted(self.log_dir.glob("*.log"))
        for path in files:
            stream = "stderr" if path.name.endswith(".stderr.log") else "stdout"
            source = path.name.rsplit(".", 2)[0]
            for line in self._tail_file(path, limit=max(limit, 1)):
                records.append(
                    {
                        "source": source,
                        "stream": stream,
                        "message": line,
                        "path": str(path),
                    }
                )
        return records[-max(limit, 1) :]

    @staticmethod
    def _tail_file(path: Path, *, limit: int) -> list[str]:
        lines: deque[str] = deque(maxlen=max(limit, 1))
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                line = raw_line.rstrip()
                if line != "":
                    lines.append(line)
        return list(lines)
