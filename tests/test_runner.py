from ai_ped_red_team.run.runner import RunExecutionConfig


def test_run_execution_config_defaults():
    config = RunExecutionConfig()
    assert config.counterbalance is True
