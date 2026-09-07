"""시뮬레이터 검증 (ADR-001, ADR-002).

원본 GMSC 대신 소형 합성 프레임을 써서 빠르게 돈다.
상관계수 검증만 실제 규모에 가까운 표본(30,000행)을 사용한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data import simulator as sim


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


def _fake_gmsc(n: int, seed: int = 7) -> pd.DataFrame:
    """GMSC 와 같은 컬럼 구성의 합성 프레임."""
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 90, size=n)
    income = rng.lognormal(8.4, 0.7, size=n)
    income[rng.random(n) < 0.2] = np.nan          # 원본의 MonthlyIncome 결측 재현
    target = (rng.random(n) < 0.067).astype(int)  # 부도율 약 6.7%
    return pd.DataFrame({
        "SeriousDlqin2yrs": target,
        "RevolvingUtilizationOfUnsecuredLines": rng.random(n),
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": rng.integers(0, 4, size=n),
        "DebtRatio": rng.random(n),
        "MonthlyIncome": income,
        "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 20, size=n),
        "NumberOfTimes90DaysLate": rng.integers(0, 3, size=n),
        "NumberRealEstateLoansOrLines": rng.integers(0, 5, size=n),
        "NumberOfTime60-89DaysPastDueNotWorse": rng.integers(0, 3, size=n),
        "NumberOfDependents": rng.integers(0, 5, size=n).astype(float),
    })


@pytest.fixture(scope="module")
def simulated(config) -> pd.DataFrame:
    return sim.simulate(_fake_gmsc(30_000), config)


# --- 상관계수 범위 ---------------------------------------------------------

def test_alternative_feature_correlations(simulated, config):
    """대안 5변수는 target 과 절댓값 0.3~0.5 의 상관을 가져야 한다 (ADR-001)."""
    lo = config["simulator"]["target_correlation"]["min"]
    hi = config["simulator"]["target_correlation"]["max"]
    target = simulated[config["data"]["target_column"]]
    for feature in config["simulator"]["alternative_features"]:
        assert feature in simulated.columns
        corr = float(np.corrcoef(simulated[feature], target)[0, 1])
        assert lo <= abs(corr) <= hi, f"{feature}: corr={corr:.4f} 가 [{lo}, {hi}] 밖"
        assert corr < 0, f"{feature}: 우량 행동 지표이므로 부도와 음의 상관이어야 한다 (corr={corr:.4f})"


def test_alternative_features_are_not_target_copies(simulated, config):
    """상관 상한을 넘는 누수성 피처가 없어야 한다."""
    target = simulated[config["data"]["target_column"]]
    for feature in config["simulator"]["alternative_features"]:
        assert abs(float(np.corrcoef(simulated[feature], target)[0, 1])) < 0.7


# --- 씬파일러 -------------------------------------------------------------

@pytest.mark.parametrize("ratio", [0.0, 0.1, 0.3, 0.5])
def test_thin_filer_ratio(config, ratio):
    """is_thin_filer 비율이 thin_filer_ratio 와 일치해야 한다."""
    df = sim.add_demographics(_fake_gmsc(10_000), config)
    df = sim.add_alternative_features(df, config)
    df = sim.apply_thin_filer(df, config, thin_filer_ratio=ratio)
    flag = config["simulator"]["thin_filer_flag_column"]
    assert abs(df[flag].mean() - ratio) < 1e-3


def test_thin_filer_masks_delinquency_columns(simulated, config):
    """씬파일러는 연체 3컬럼이 전부 결측, 비씬파일러는 결측 없음."""
    masked = config["simulator"]["thin_filer_masked_columns"]
    flag = config["simulator"]["thin_filer_flag_column"]
    thin = simulated[simulated[flag] == 1]
    non_thin = simulated[simulated[flag] == 0]
    for col in masked:
        assert thin[col].isna().all()
        assert not non_thin[col].isna().any()


def test_thin_filer_flag_follows_spec_definition(simulated, config):
    """is_thin_filer 정의: 연체 관련 컬럼 2개 이상 결측."""
    masked = config["simulator"]["thin_filer_masked_columns"]
    threshold = config["simulator"]["thin_filer_min_missing"]
    flag = config["simulator"]["thin_filer_flag_column"]
    expected = (simulated[masked].isna().sum(axis=1) >= threshold).astype(int)
    pd.testing.assert_series_equal(simulated[flag], expected, check_names=False)


def test_thin_filer_noise_increases_dispersion(config):
    """노이즈 주입 컬럼의 분산이 씬파일러 쪽에서 더 커야 한다 (정보량 축소)."""
    df = sim.add_demographics(_fake_gmsc(20_000), config)
    df = sim.add_alternative_features(df, config)
    df = sim.apply_thin_filer(df, config, thin_filer_ratio=0.3)
    flag = config["simulator"]["thin_filer_flag_column"]
    for col in config["simulator"]["thin_filer_noised_columns"]:
        thin_sd = df.loc[df[flag] == 1, col].std()
        non_thin_sd = df.loc[df[flag] == 0, col].std()
        assert thin_sd > non_thin_sd, f"{col}: 씬파일러 분산이 더 크지 않다"


# --- 편향 주입 -------------------------------------------------------------

def test_bias_direction(config):
    """bias_ratio 가 커질수록 female / 55+ 의 대상 피처 평균이 단조 감소해야 한다."""
    base = _fake_gmsc(30_000)
    disadvantaged = config["fairness"]["disadvantaged_groups"]
    features = config["simulator"]["biased_features"]

    means: dict[float, dict[str, dict[str, float]]] = {}
    for ratio in (0.0, 0.1, 0.5):
        df = sim.simulate(base, config, bias_ratio=ratio)
        means[ratio] = {
            feature: {
                "female": df.loc[df["gender"] == disadvantaged["gender"], feature].mean(),
                "male": df.loc[df["gender"] != disadvantaged["gender"], feature].mean(),
                "old": df.loc[df["age_group"] == disadvantaged["age_group"], feature].mean(),
            }
            for feature in features
        }

    for feature in features:
        assert means[0.5][feature]["female"] < means[0.1][feature]["female"] < means[0.0][feature]["female"]
        assert means[0.5][feature]["old"] < means[0.1][feature]["old"] < means[0.0][feature]["old"]
        # 편향 없는 상태에서는 성별 격차가 없어야 한다(gender 는 무작위 합성).
        assert abs(means[0.0][feature]["female"] - means[0.0][feature]["male"]) < 0.01
        # 편향 주입 후에는 female 이 male 보다 낮아야 한다.
        assert means[0.5][feature]["female"] < means[0.5][feature]["male"]


def test_bias_shift_magnitude(config):
    """하향폭은 bias_ratio × bias_shift_per_ratio 여야 한다."""
    base = _fake_gmsc(30_000)
    ratio = 0.5
    shift = ratio * config["simulator"]["bias_shift_per_ratio"]
    clean = sim.simulate(base, config, bias_ratio=0.0)
    biased = sim.simulate(base, config, bias_ratio=ratio)
    feature = config["simulator"]["biased_features"][0]
    # 남성 & 55+ 아닌 집단은 변화가 없어야 한다.
    untouched = (clean["gender"] == "male") & (clean["age_group"] != "55+")
    assert np.allclose(clean.loc[untouched, feature], biased.loc[untouched, feature])
    # female & 55+ 아닌 집단은 정확히 shift 만큼 하향.
    only_female = (clean["gender"] == "female") & (clean["age_group"] != "55+")
    diff = clean.loc[only_female, feature] - biased.loc[only_female, feature]
    assert np.allclose(diff, shift, atol=1e-9)


def test_target_is_never_modified(config):
    """ADR-002: 편향 주입은 타겟을 건드리지 않는다."""
    base = _fake_gmsc(10_000)
    target_col = config["data"]["target_column"]
    for ratio in (0.0, 0.1, 0.5):
        df = sim.simulate(base, config, bias_ratio=ratio)
        pd.testing.assert_series_equal(df[target_col], base[target_col], check_names=False)


# --- 인구통계 -------------------------------------------------------------

def test_demographics(config):
    """gender 는 대략 50:50, age_group 은 결측 없이 세 구간에 배정된다."""
    df = sim.add_demographics(_fake_gmsc(20_000), config)
    gender_col = config["simulator"]["gender_column"]
    age_group_col = config["simulator"]["age_group_column"]

    shares = df[gender_col].value_counts(normalize=True)
    assert set(shares.index) == set(config["fairness"]["groups"]["gender"])
    assert abs(shares["female"] - 0.5) < 0.02

    assert not df[age_group_col].isna().any()
    assert set(df[age_group_col].unique()) <= set(config["fairness"]["groups"]["age_group"])
    # 실제 age 기반인지: 각 구간의 age 범위가 겹치지 않아야 한다.
    assert df.loc[df[age_group_col] == "20-34", "age"].max() < 35
    assert df.loc[df[age_group_col] == "35-54", "age"].min() >= 35
    assert df.loc[df[age_group_col] == "55+", "age"].min() >= 55


def test_gender_independent_of_target(simulated, config):
    """합성 성별은 타겟과 독립이어야 한다(편향은 피처에만 주입)."""
    target = simulated[config["data"]["target_column"]]
    is_female = (simulated["gender"] == "female").astype(int)
    assert abs(float(np.corrcoef(is_female, target)[0, 1])) < 0.02
