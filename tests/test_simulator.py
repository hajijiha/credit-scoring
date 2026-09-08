"""시뮬레이터 검증 (ADR-001, ADR-002).

원본 GMSC 대신 소형 합성 프레임을 써서 빠르게 돈다.
상관계수 검증만 실제 규모에 가까운 표본(30,000행)을 사용한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathlib import Path

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


# 합성 표본은 실제 GMSC 보다 상관 여유가 작다(소득·연령 분포가 다르다).
# 배포 설정(bias_ratio=0.3)의 상관 검증은 실제 산출물로 하고(test_delivered_dataset_correlations),
# 합성 표본에서는 여유가 있는 수준으로 확인한다.
FIXTURE_BIAS_RATIO = 0.1


@pytest.fixture(scope="module")
def simulated(config) -> pd.DataFrame:
    return sim.simulate(_fake_gmsc(30_000), config, bias_ratio=FIXTURE_BIAS_RATIO)


# --- 상관계수 범위 ---------------------------------------------------------

def test_delivered_dataset_correlations(config):
    """**실제 산출물**의 상관이 SPEC §4 범위 안에 있어야 한다.

    배포되는 설정(config 의 bias_ratio)으로 만든 data/processed/train.parquet 을 검사한다.
    합성 표본이 아니라 이 파일이 리포트에 실리는 값이다.
    """
    path = (
        Path(__file__).resolve().parents[1]
        / config["preprocess"]["output_dir"] / "train.parquet"
    )
    if not path.exists():
        pytest.skip(f"{path} 없음 — 먼저 python -m src.data.preprocessor 실행")
    frame = pd.read_parquet(path)
    lo = config["simulator"]["target_correlation"]["min"]
    hi = config["simulator"]["target_correlation"]["max"]
    target = frame[config["data"]["target_column"]]
    margins = {}
    for feature in config["simulator"]["alternative_features"]:
        corr = abs(float(np.corrcoef(frame[feature], target)[0, 1]))
        assert lo <= corr <= hi, f"{feature}: |corr|={corr:.4f} 가 [{lo}, {hi}] 밖"
        margins[feature] = corr - lo
    # 하한 여유가 0 에 붙어 있으면 설정 변경 시 쉽게 깨진다. 기록만 남긴다.
    assert min(margins.values()) >= 0.0, margins


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


@pytest.mark.parametrize("bias_ratio", [0.0, 0.1, 0.2])
def test_correlations_hold_across_bias_ratio(config, bias_ratio):
    """편향 주입은 대상 피처의 상관을 낮춘다.

    ADR-001 v2 로 생성 상관이 0.31~0.35 로 낮아졌으므로 여유가 줄었다.
    GMSC 실데이터 기준 유효 bias_ratio 범위는 0 ~ 0.3 이며 0.4 부터 telecom/utility 가
    0.30 아래로 내려간다. 이 테스트의 합성 표본은 여유가 더 적어 0.2 까지만 확인한다.
    """
    lo = config["simulator"]["target_correlation"]["min"]
    hi = config["simulator"]["target_correlation"]["max"]
    df = sim.simulate(_fake_gmsc(30_000), config, bias_ratio=bias_ratio)
    target = df[config["data"]["target_column"]]
    for feature in config["simulator"]["alternative_features"]:
        corr = abs(float(np.corrcoef(df[feature], target)[0, 1]))
        assert lo <= corr <= hi, f"bias_ratio={bias_ratio} {feature}: |corr|={corr:.4f}"


def test_shared_latent_factor_creates_within_class_correlation(simulated, config):
    """ADR-001 v2: 타겟을 조건으로 해도 대안 피처들이 서로 상관을 가져야 한다.

    조건부 독립이면 다섯 변수의 정보가 그대로 합산돼 결합 판별력이 비현실적으로 커진다(v1 문제).
    """
    features = config["simulator"]["alternative_features"]
    target = simulated[config["data"]["target_column"]]
    means = []
    for cls in (0, 1):
        block = simulated.loc[target == cls, features]
        corr = block.corr().to_numpy()
        means.append(corr[np.triu_indices(len(features), 1)].mean())
    for value in means:
        assert value > 0.4, f"클래스 내부 평균 상관 {value:.3f} — 공유 잠재요인이 약하다"


def test_joint_separation_is_bounded(simulated, config):
    """대안 5변수 결합 판별력이 과도하지 않아야 한다 (ADR-001 v2 누수 회귀 방지)."""
    features = config["simulator"]["alternative_features"]
    y = simulated[config["data"]["target_column"]].to_numpy()
    x = simulated[features].to_numpy()
    m0, m1 = x[y == 0].mean(0), x[y == 1].mean(0)
    w = np.linalg.solve(np.cov(x[y == 0].T), m1 - m0)
    score = x @ w
    a, b = score[y == 0], score[y == 1]
    d = abs(b.mean() - a.mean()) / np.sqrt(0.5 * (a.var() + b.var()))
    assert d < 3.0, f"결합 효과크기 d={d:.2f} — v1 수준(3.96)의 누수가 재발했다"


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
    """ADR-010: 연체 결측은 missing_only / both 유형에만 적용된다."""
    sim_cfg = config["simulator"]
    masked = sim_cfg["thin_filer_masked_columns"]
    type_col = sim_cfg["thin_filer_type_column"]
    should_be_missing = simulated[type_col].isin(["missing_only", "both"])
    for col in masked:
        assert simulated.loc[should_be_missing, col].isna().all()
        assert not simulated.loc[~should_be_missing, col].isna().any()


def test_thin_filer_flag_follows_spec_definition(simulated, config):
    """is_thin_filer 정의 (SPEC §4): 연체 컬럼 2개 이상 결측 OR 거래 이력 12개월 미만."""
    sim_cfg = config["simulator"]
    masked = sim_cfg["thin_filer_masked_columns"]
    hist = sim_cfg["credit_history"]
    flag = sim_cfg["thin_filer_flag_column"]
    by_missing = simulated[masked].isna().sum(axis=1) >= sim_cfg["thin_filer_min_missing"]
    by_history = simulated[hist["column"]] < hist["min_months_for_normal"]
    expected = (by_missing | by_history).astype(int)
    pd.testing.assert_series_equal(simulated[flag], expected, check_names=False)


def test_credit_history_months_ranges(simulated, config):
    """ADR-009 / ADR-010: 이력 부족 유형만 3~11개월, 나머지는 12~120개월."""
    sim_cfg = config["simulator"]
    hist = sim_cfg["credit_history"]
    col = hist["column"]
    type_col = sim_cfg["thin_filer_type_column"]
    short = simulated[type_col].isin(["history_only", "both"])
    assert simulated.loc[short, col].between(*hist["thin_range"]).all()
    assert simulated.loc[~short, col].between(*hist["normal_range"]).all()
    assert (simulated.loc[short, col] < hist["min_months_for_normal"]).all()
    assert (simulated.loc[~short, col] >= hist["min_months_for_normal"]).all()


def test_thin_filer_type_mix(simulated, config):
    """ADR-010: 씬파일러 내부 유형 비중이 설정과 일치해야 한다."""
    sim_cfg = config["simulator"]
    type_col = sim_cfg["thin_filer_type_column"]
    flag = sim_cfg["thin_filer_flag_column"]
    mix = sim_cfg["thin_filer_type_mix"]

    thin = simulated[simulated[flag] == 1]
    assert set(simulated[type_col].unique()) <= {"none", *mix}
    shares = thin[type_col].value_counts(normalize=True)
    for kind, expected in mix.items():
        assert abs(shares[kind] - expected) < 0.02, f"{kind}: {shares[kind]:.3f} vs {expected}"
    # 비씬파일러는 전부 none
    assert (simulated.loc[simulated[flag] == 0, type_col] == "none").all()


def test_thin_filer_conditions_are_not_identical(simulated, config):
    """ADR-010: 두 판정 조건이 서로 다른 집합이어야 OR 정의가 의미를 갖는다."""
    sim_cfg = config["simulator"]
    masked = sim_cfg["thin_filer_masked_columns"]
    hist = sim_cfg["credit_history"]
    by_missing = simulated[masked].isna().sum(axis=1) >= sim_cfg["thin_filer_min_missing"]
    by_history = simulated[hist["column"]] < hist["min_months_for_normal"]
    intersection = int((by_missing & by_history).sum())
    union = int((by_missing | by_history).sum())
    assert intersection < union, "두 조건이 동일 집합이면 OR 정의가 무의미해진다"
    assert int(by_missing.sum()) < union
    assert int(by_history.sum()) < union


def test_credit_history_weak_positive_age_correlation(simulated, config):
    """ADR-009: age 와 약한 양의 상관. 강하면 age 의 복제가 된다."""
    col = config["simulator"]["credit_history"]["column"]
    corr = float(np.corrcoef(simulated[col], simulated["age"])[0, 1])
    assert 0.05 < corr < 0.60, f"age 상관 {corr:.4f} 가 '약한 양의 상관' 범위를 벗어남"


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
                "disadvantaged_age": df.loc[df["age_group"] == disadvantaged["age_group"], feature].mean(),
                "other_age": df.loc[df["age_group"] != disadvantaged["age_group"], feature].mean(),
            }
            for feature in features
        }

    for feature in features:
        assert means[0.5][feature]["female"] < means[0.1][feature]["female"] < means[0.0][feature]["female"]
        assert (means[0.5][feature]["disadvantaged_age"]
                < means[0.1][feature]["disadvantaged_age"]
                < means[0.0][feature]["disadvantaged_age"])
        # 불이익 집단과 나머지 연령대의 격차가 bias_ratio 와 함께 벌어져야 한다 (ADR-002 v2).
        gaps = [means[r][feature]["other_age"] - means[r][feature]["disadvantaged_age"]
                for r in (0.0, 0.1, 0.5)]
        assert gaps[0] < gaps[1] < gaps[2], f"{feature}: 연령대 격차가 단조 증가하지 않는다 {gaps}"
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
    disadvantaged = config["fairness"]["disadvantaged_groups"]
    # 비불이익 집단(남성 & 불이익 연령대 아님)은 변화가 없어야 한다.
    untouched = (clean["gender"] != disadvantaged["gender"]) & (clean["age_group"] != disadvantaged["age_group"])
    assert np.allclose(clean.loc[untouched, feature], biased.loc[untouched, feature])
    # 불이익 성별만 해당하는 집단은 정확히 shift 만큼 하향.
    only_female = (clean["gender"] == disadvantaged["gender"]) & (clean["age_group"] != disadvantaged["age_group"])
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
