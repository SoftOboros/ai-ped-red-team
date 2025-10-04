from hypothesis import given
from hypothesis import strategies as st

from ai_ped_red_team.normalize.textnorm import directive_ratio, normalize_text


@given(st.text())
def test_normalize_roundtrip(text: str):
    result = normalize_text(text)
    assert isinstance(result["text"], str)
    assert isinstance(result["tokens"], list)


@given(st.text())
def test_directive_ratio_bounds(text: str):
    ratio = directive_ratio(text)
    assert 0.0 <= ratio <= 1.0
