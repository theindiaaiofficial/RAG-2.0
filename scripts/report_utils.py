from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def update_report_section(title: str, body: str) -> None:
    path = ROOT / "TEST_REPORT.md"
    current = path.read_text(encoding="utf-8") if path.exists() else "# Verification Report\n"
    marker = f"## {title}"
    section = f"{marker}\n\n**Run time:** {datetime.now(timezone.utc).isoformat()}\n\n{body.strip()}\n"
    if marker in current:
        before, remainder = current.split(marker, 1)
        next_section = remainder.find("\n## ")
        current = before + section + (remainder[next_section + 1 :] if next_section >= 0 else "")
    else:
        current = current.rstrip() + "\n\n" + section
    path.write_text(current, encoding="utf-8")
