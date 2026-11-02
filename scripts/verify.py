from __future__ import annotations

import importlib.util
import re
from datetime import datetime, timezone
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(label: str, command: list[str]) -> tuple[bool, str]:
    print(f"\n== {label} ==")

    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )

        output = ((result.stdout or "") + (result.stderr or "")).strip()

        print(output or "Completed.")

        return result.returncode == 0, output

    except Exception as exc:
        output = f"{type(exc).__name__}: {exc}"
        print(output)
        return False, output


def main() -> int:
    checks: list[tuple[str, bool, str]] = []
    checks.append(("Python version", sys.version_info >= (3, 11), sys.version.split()[0]))

    for module, label in [
        ("fastapi", "FastAPI"),
        ("pydantic", "Pydantic"),
        ("litellm", "LiteLLM"),
        ("chromadb", "ChromaDB"),
        ("google.genai", "Google GenAI SDK"),
        ("ruff", "Ruff"),
        ("pypdf", "pypdf"),
        ("docx", "python-docx"),
    ]:
        try:
            present = importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            present = False
        checks.append(
            (
                f"Dependency: {label}",
                present,
                "installed" if present else "not installed",
            )
        )

    compile_ok, compile_out = run(
        "Compile all Python modules",
        [sys.executable, "-m", "compileall", "-q", "app", "scripts", "tests"],
    )
    checks.append(("Python compilation", compile_ok, compile_out or "compiled"))

    test_ok, test_out = run("Pytest suite", [sys.executable, "-m", "pytest", "-q", "tests"])
    checks.append(("Pytest suite", test_ok, test_out))

    if importlib.util.find_spec("ruff") is not None:
        lint_ok, lint_out = run(
            "Ruff lint",
            [sys.executable, "-m", "ruff", "check", "app", "scripts", "tests"],
        )
    else:
        lint_ok, lint_out = False, "Ruff is not installed; lint check blocked."
        print(f"\n== Ruff lint ==\n{lint_out}")
    checks.append(("Ruff lint", lint_ok, lint_out))

    print("\nPROJECT VERIFICATION SUMMARY")
    passed = failed = 0
    skipped = (
        int(re.search(r"(\d+) skipped", test_out).group(1)) if re.search(r"(\d+) skipped", test_out) else 0
    )
    for label, ok, detail in checks:
        if ok:
            passed += 1
            marker = "PASS"
        else:
            failed += 1
            marker = "BLOCKED/FAIL"
        print(f"[{marker}] {label}: {detail.splitlines()[-1][:240] if detail else ''}")
    print(f"\nChecks passed: {passed}; blocked/failed: {failed}; skipped: {skipped}")
    print("Live provider/model checks are separate: python scripts/verify_live.py")

    report_lines = [
        f"**Run time:** {datetime.now(timezone.utc).isoformat()}",
        "",
        f"**Offline test result:** {test_out.splitlines()[-1] if test_out else 'No pytest output'}",
        f"**Compilation:** {'PASS' if compile_ok else 'FAIL'}",
        f"**Ruff:** {'PASS' if lint_ok else 'BLOCKED/FAIL — ' + lint_out}",
        f"**Verification checks:** {passed} passed; {failed} blocked/failed; {skipped} pytest skipped.",
        "",
        "### Dependency checks",
    ]
    for label, ok, detail in checks:
        if label.startswith("Dependency:"):
            report_lines.append(
                f"- [{'PASS' if ok else 'BLOCKED'}] {label.removeprefix('Dependency: ')} — {detail}"
            )
    report_lines += [
        "",
        "### Notes",
        "- Live LLM verification is separate and must be run with `python scripts/verify_live.py`.",
        "- Dependency checks marked BLOCKED were not available in this runtime; related real-runtime behaviors were not claimed as passing.",
    ]
    report_path = ROOT / "TEST_REPORT.md"
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else "# Verification Report\n"
    marker = "## Offline verification"
    section = marker + "\n\n" + "\n".join(report_lines) + "\n"
    if marker in report:
        before, rest = report.split(marker, 1)
        next_section = rest.find("\n## ")
        report = before + section + (rest[next_section + 1 :] if next_section >= 0 else "")
    else:
        report = report.rstrip() + "\n\n" + section
    report_path.write_text(report, encoding="utf-8")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
