import json

from ai_ped_red_team.templates.loader import load_template


def test_load_template_reads_json(tmp_path):
    template_path = tmp_path / "template.json"
    template_path.write_text(
        json.dumps(
            {
                "id": "demo",
                "title": "Test",
                "hotness": "cold",
                "template_text": "{{prompt}}",
                "placeholders": [],
                "metadata": {},
            }
        )
    )
    template = load_template(template_path)
    assert template.id == "demo"
