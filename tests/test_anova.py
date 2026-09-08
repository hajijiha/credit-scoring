"""ANOVA 모듈 검증 (SPEC §5, ADR-011).

알려진 값으로 F 통계량·η² 의 정확성을 확인한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.config import load_config
from src.statistics.anova import (
    eta_squared,
    format_report,
    homogeneity_check,
    interpret_eta_squared,
    normality_check,
    run_test,
    tukey_hsd,
)


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


# --- F 통계량 -------------------------------------------------------------

def test_f_statistic_known_values():
    """손으로 계산 가능한 예제. 세 집단 [1,2,3] / [4,5,6] / [7,8,9].

    집단평균 2, 5, 8 / 전체평균 5.
    SS_between = 3*(9+0+9) = 54, df=2 -> MS_between = 27
    SS_within  = 3*2 = 6,      df=6 -> MS_within  = 1
    F = 27
    """
    groups = [np.array([1.0, 2, 3]), np.array([4.0, 5, 6]), np.array([7.0, 8, 9])]
    f_stat, _ = stats.f_oneway(*groups)
    assert f_stat == pytest.approx(27.0)


def test_eta_squared_known_values():
    """같은 예제: η² = SS_between / SS_total = 54 / 60 = 0.9."""
    groups = [np.array([1.0, 2, 3]), np.array([4.0, 5, 6]), np.array([7.0, 8, 9])]
    assert eta_squared(groups) == pytest.approx(0.9)


def test_eta_squared_no_group_difference():
    """집단 평균이 모두 같으면 η² = 0."""
    groups = [np.array([1.0, 2, 3]), np.array([1.0, 2, 3]), np.array([3.0, 2, 1])]
    assert eta_squared(groups) == pytest.approx(0.0, abs=1e-12)


def test_eta_squared_no_within_variance():
    """집단 내 분산이 0이면 η² = 1."""
    groups = [np.full(5, 1.0), np.full(5, 2.0), np.full(5, 3.0)]
    assert eta_squared(groups) == pytest.approx(1.0)


def test_eta_squared_unequal_group_sizes():
    """가중치가 집단 크기에 비례해야 한다."""
    groups = [np.array([1.0] * 10), np.array([3.0] * 2)]
    # grand mean = (10*1 + 2*3)/12 = 4/3
    # SS_between = 10*(1-4/3)^2 + 2*(3-4/3)^2 = 10/9 + 50/9 = 60/9
    # SS_total 도 동일(집단 내 분산 0) -> 1.0
    assert eta_squared(groups) == pytest.approx(1.0)


# --- 효과크기 해석 ---------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [(0.005, "Negligible"), (0.01, "Small (weak)"), (0.05, "Small (weak)"),
     (0.06, "Medium (moderate)"), (0.13, "Medium (moderate)"),
     (0.14, "Large (strong)"), (0.99, "Large (strong)")],
)
def test_interpret_eta_squared(config, value, expected):
    """SPEC §5 의 0.01 / 0.06 / 0.14 기준."""
    assert interpret_eta_squared(value, config["anova"]["eta_squared_thresholds"]) == expected


# --- 진단 검정 -------------------------------------------------------------

def test_normality_detects_normal_data():
    rng = np.random.default_rng(0)
    groups = {f"g{i}": rng.normal(size=40) for i in range(3)}
    assert normality_check(groups)["all_normal"]


def test_normality_detects_non_normal_data():
    rng = np.random.default_rng(0)
    groups = {"skewed": rng.exponential(size=200), "normal": rng.normal(size=200)}
    result = normality_check(groups)
    assert not result["all_normal"]
    assert not result["per_group"]["skewed"]["normal"]


def test_homogeneity_detects_equal_and_unequal_variance():
    rng = np.random.default_rng(1)
    equal = {"a": rng.normal(0, 1, 200), "b": rng.normal(5, 1, 200)}
    assert homogeneity_check(equal)["equal_variance"]
    unequal = {"a": rng.normal(0, 1, 200), "b": rng.normal(0, 6, 200)}
    assert not homogeneity_check(unequal)["equal_variance"]


# --- Tukey HSD ------------------------------------------------------------

def test_tukey_finds_the_separated_group():
    """두 집단은 같고 하나만 떨어져 있으면 그 쌍만 유의해야 한다."""
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "auc": np.concatenate([rng.normal(0.80, 0.01, 30),
                               rng.normal(0.80, 0.01, 30),
                               rng.normal(0.90, 0.01, 30)]),
        "grp": ["a"] * 30 + ["b"] * 30 + ["c"] * 30,
    })
    rows = {r["comparison"]: r for r in tukey_hsd(df, "grp", 0.05)}
    assert not rows["a vs b"]["significant"]
    assert rows["a vs c"]["significant"]
    assert rows["b vs c"]["significant"]


# --- 통제(ADR-011) --------------------------------------------------------

def _fake_folds() -> pd.DataFrame:
    """18조합 × 20 fold 구조의 합성 fold AUC."""
    rng = np.random.default_rng(3)
    base = {"traditional_only": 0.83, "alternative_only": 0.89, "combined": 0.94}
    bump = {"logistic_regression": -0.03, "xgboost": 0.0, "lightgbm": -0.001}
    rows = []
    for group, level in base.items():
        for algo, delta in bump.items():
            for imbalance in ("class_weight", "smote"):
                for fold in range(20):
                    rows.append({
                        "algorithm": algo, "feature_group": group, "imbalance": imbalance,
                        "fold": fold,
                        "auc": level + delta + (0.0 if imbalance == "class_weight" else -0.005)
                               + rng.normal(0, 0.003),
                    })
    return pd.DataFrame(rows)


def test_controlled_scope_uses_twenty_observations(config):
    """ADR-011 / ADR-005: 통제하면 그룹당 n=20 이어야 한다."""
    spec = config["anova"]["tests"]["test_1_feature_group"]
    result = run_test(_fake_folds(), spec, config, scope="controlled")
    assert set(result["group_n"].values()) == {20}
    assert result["control"] == spec["control"]


def test_pooled_scope_uses_all_observations(config):
    """참고치(pooled)는 통제 없이 전체를 쓴다."""
    spec = config["anova"]["tests"]["test_1_feature_group"]
    result = run_test(_fake_folds(), spec, config, scope="pooled")
    assert set(result["group_n"].values()) == {120}
    assert result["control"] == {}


def test_significant_test_includes_tukey(config):
    """p < alpha 이면 사후검정이 포함돼야 한다."""
    spec = config["anova"]["tests"]["test_1_feature_group"]
    result = run_test(_fake_folds(), spec, config)
    assert result["is_significant"]
    assert "post_hoc_tukey" in result
    assert len(result["post_hoc_tukey"]) == 3


def test_non_significant_test_has_no_tukey(config):
    """집단 차이가 없으면 유의하지 않고 사후검정도 없어야 한다."""
    rng = np.random.default_rng(4)
    folds = _fake_folds()
    folds["auc"] = rng.normal(0.9, 0.003, len(folds))     # 집단 구분 제거
    spec = config["anova"]["tests"]["test_1_feature_group"]
    result = run_test(folds, spec, config)
    assert not result["is_significant"]
    assert "post_hoc_tukey" not in result


def test_extreme_effect_size_note(config):
    """η² > 임계값이면 해석 주의 문구가 붙어야 한다."""
    spec = config["anova"]["tests"]["test_1_feature_group"]
    result = run_test(_fake_folds(), spec, config)
    assert result["effect_size_eta_squared"] > config["anova"]["eta_squared_extreme"]
    assert "effect_size_note" in result


def test_report_renders(config):
    """리포트 문자열에 필수 항목이 들어가야 한다."""
    spec = config["anova"]["tests"]["test_1_feature_group"]
    text = format_report({"test_1": run_test(_fake_folds(), spec, config)})
    for token in ("F 통계량", "p-value", "η²", "Shapiro-Wilk", "Levene", "Tukey HSD"):
        assert token in text
