from ai_ped_red_team.models import gateway


def test_llm_complete_callable():
    assert hasattr(gateway, 'llm_complete')
