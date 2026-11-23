"""Record selected IDs and outcomes without changes to test execution."""

import json
from pathlib import Path


_reports = {}


def pytest_runtest_logreport(report):
    _reports.setdefault(report.nodeid, []).append({
        "phase": report.when,
        "outcome": report.outcome,
    })


def pytest_sessionfinish(session, exitstatus):
    selected = [item.nodeid for item in session.items]
    if selected:
        name = Path(selected[0].split("::", 1)[0]).stem
        record = {
            "exit_code": int(exitstatus),
            "selected": selected,
            "reports": _reports,
        }
        destination = Path(__file__).parent / "current-results"
        destination.mkdir(exist_ok=True)
        (destination / f"{name}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
