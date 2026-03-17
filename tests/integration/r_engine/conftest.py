import numpy as np
import pandas as pd
import pytest

try:
    from app.core.engine.environment import get_r_environment
    from app.core.engine.model import GAMLSS, FittedGAMLSSModel

    r_env = get_r_environment()
    R_AVAILABLE = True
except Exception:
    R_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not R_AVAILABLE, reason="R environment with gamlss packages not available"
)
seed = np.random.default_rng(42)


@pytest.fixture
def synthetic_normal_data() -> pd.DataFrame:
    patients_age = seed.integers(5, 80, size=100)
    volumes = [1000 + 2 * age + seed.normal(0, 30) for age in patients_age]

    return pd.DataFrame({"Age": patients_age, "Volume": volumes})
