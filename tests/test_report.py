from pathlib import Path

from ai_ped_red_team.analyze.report import render_report


def test_render_report(tmp_path: Path):
    summary = tmp_path / "summary.json"
    summary.write_text('{"metrics": {}, "by_student": {}, "tests": {}}')
    outputs = render_report(summary)
    assert len(outputs) == 2
    for path in outputs:
        assert path.exists()
