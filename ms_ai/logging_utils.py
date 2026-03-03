"""
Lightweight daily CSV logger for CLI command runs.

Appends one line per invocation to ``logs/daily_log.csv``.
Creates the file (with header) on first use.
"""

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

_LOG_DIR = Path("logs")
_LOG_FILE = _LOG_DIR / "daily_log.csv"
_TZ_TR = ZoneInfo("Europe/Istanbul")

_FIELDS = [
    "timestamp_utc",
    "timestamp_tr",
    "command",
    "use",
    "split",
    "rows_written",
    "output_path",
    "status",
]


def log_run(
    command: str,
    *,
    use: Optional[str] = None,
    split: Optional[str] = None,
    rows_written: Optional[int] = None,
    output_path: Optional[str] = None,
    status: str = "OK",
) -> None:
    """Append a single row to the daily CSV log."""
    _LOG_DIR.mkdir(exist_ok=True)
    write_header = not _LOG_FILE.exists()

    now_utc = datetime.now(timezone.utc)
    now_tr = now_utc.astimezone(_TZ_TR)

    with open(_LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp_utc": now_utc.isoformat(),
            "timestamp_tr": now_tr.isoformat(),
            "command": command,
            "use": use or "",
            "split": split or "",
            "rows_written": rows_written if rows_written is not None else "",
            "output_path": output_path or "",
            "status": status,
        })
