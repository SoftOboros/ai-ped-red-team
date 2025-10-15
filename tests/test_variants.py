from ai_ped_red_team.generate import variants


def test_fallback_variants_retain_placeholders():
    items = variants._fallback_variants("cold", n=3, seed=0)
    for item in items:
        assert "{{STUDENT_NAME}}" in item.variant_prompt
        assert "{{SUPPORT_NEED}}" in item.variant_prompt
