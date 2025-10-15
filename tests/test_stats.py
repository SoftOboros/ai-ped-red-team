import pandas as pd

from ai_ped_red_team.analyze.stats import summarize_stats


def test_summarize_stats_returns_dict():
    df = pd.DataFrame(columns=["variant_id", "student", "response"])
    summary = summarize_stats(df)
    assert "metrics" in summary
