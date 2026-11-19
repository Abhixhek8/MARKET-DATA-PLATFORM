from datetime import date

import pytest

from mdp.ingestion import fetch_synthetic
from mdp.transform import clean


@pytest.fixture
def sample():
    frames = [fetch_synthetic(t, date(2023, 1, 1), date(2024, 6, 30)) for t in ("AAA", "BBB", "CCC")]
    import pandas as pd
    return clean(pd.concat(frames, ignore_index=True))
