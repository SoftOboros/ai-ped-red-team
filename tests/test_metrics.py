import pandas as pd

from ai_ped_red_team.analyze.metrics import compute_metrics


def test_compute_metrics_handles_empty(tmp_path):
    data = tmp_path / 'results.jsonl'
    data.write_text('')
    df = compute_metrics(data)
    assert isinstance(df, pd.DataFrame)
    assert df.empty
