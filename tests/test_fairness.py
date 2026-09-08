"""공정성 지표 · Mitigation 검증 (SPEC §4, ADR-004)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from src.config import load_config
from src.fairness.metrics import (
    approval_mask,
    disparate_impact,
    equalized_odds,
    evaluate,
    group_rates,
)
from src.fairness.mitigation import adjusted_thresholds, compare, fit_threshold_optimizer


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


def _gender(n_female: int, n_male: int) -> pd.Series:
    return pd.Series(["female"] * n_female + ["male"] * n_male, name="gender")


# --- 승인 마스크 ----------------------------------------------------------

def test_approval_mask():
    """부도 확률이 임계값 미만이면 승인."""
    proba = np.array([0.10, 0.29, 0.30, 0.31, 0.90])
    np.testing.assert_array_equal(
        approval_mask(proba, 0.30), np.array([True, True, False, False, False])
    )


def test_group_rates():
    proba = np.array([0.1] * 3 + [0.9] * 7 + [0.1] * 9 + [0.9])
    rates = group_rates(proba, _gender(10, 10), 0.30)
    assert rates["female"] == pytest.approx(0.3)
    assert rates["male"] == pytest.approx(0.9)


# --- DI Ratio 손계산 ------------------------------------------------------

def test_disparate_impact_hand_calculation(config):
    """female 승인 4/10=0.4, male 승인 8/10=0.8 -> DI = 0.5."""
    proba = np.array([0.1] * 4 + [0.9] * 6 + [0.1] * 8 + [0.9] * 2)
    result = disparate_impact(proba, _gender(10, 10), 0.30, "female", config)
    assert result["protected_approval_rate"] == pytest.approx(0.4)
    assert result["reference_approval_rate"] == pytest.approx(0.8)
    assert result["di_ratio"] == pytest.approx(0.5)
    assert result["di_ratio_min_max"] == pytest.approx(0.5)
    assert not result["passes"]


def test_disparate_impact_perfect_parity(config):
    """두 집단 승인율이 같으면 DI = 1.0 이고 기준을 충족한다."""
    proba = np.array([0.1] * 5 + [0.9] * 5 + [0.1] * 5 + [0.9] * 5)
    result = disparate_impact(proba, _gender(10, 10), 0.30, "female", config)
    assert result["di_ratio"] == pytest.approx(1.0)
    assert result["passes"]


def test_disparate_impact_reference_is_pooled_complement(config):
    """3집단이면 기준 집단은 나머지 전체의 합산 승인율이다."""
    # 20-34: 2/10, 35-54: 8/10, 55+: 10/10 -> 나머지 합산 18/20 = 0.9
    proba = np.concatenate([
        np.array([0.1] * 2 + [0.9] * 8),
        np.array([0.1] * 8 + [0.9] * 2),
        np.array([0.1] * 10),
    ])
    sensitive = pd.Series(["20-34"] * 10 + ["35-54"] * 10 + ["55+"] * 10, name="age_group")
    result = disparate_impact(proba, sensitive, 0.30, "20-34", config)
    assert result["protected_approval_rate"] == pytest.approx(0.2)
    assert result["reference_approval_rate"] == pytest.approx(0.9)
    assert result["di_ratio"] == pytest.approx(0.2 / 0.9)
    assert result["di_ratio_min_max"] == pytest.approx(0.2)


# --- Equalized Odds 손계산 -----------------------------------------------

def test_equalized_odds_hand_calculation(config):
    """설계한 값으로 TPR/FPR 차이를 확인한다.

    female: 부도 4명 중 3명을 거절 -> TPR 0.75 / 정상 4명 중 2명을 거절 -> FPR 0.50
    male  : 부도 4명 중 1명을 거절 -> TPR 0.25 / 정상 4명 중 0명을 거절 -> FPR 0.00
    -> TPR차 0.50, FPR차 0.50
    """
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0] * 2)
    proba = np.array([
        0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.1, 0.1,     # female
        0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,     # male
    ])
    sensitive = pd.Series(["female"] * 8 + ["male"] * 8, name="gender")
    result = equalized_odds(y, proba, sensitive, 0.30, config)
    assert result["tpr_by_group"]["female"] == pytest.approx(0.75)
    assert result["tpr_by_group"]["male"] == pytest.approx(0.25)
    assert result["fpr_by_group"]["female"] == pytest.approx(0.50)
    assert result["fpr_by_group"]["male"] == pytest.approx(0.0)
    assert result["equalized_odds_tpr_diff"] == pytest.approx(0.50)
    assert result["equalized_odds_fpr_diff"] == pytest.approx(0.50)
    assert not result["passes"]


def test_equalized_odds_passes_when_equal(config):
    """두 집단의 예측이 동일하면 차이가 0 이고 기준을 충족한다."""
    y = np.array([1, 1, 0, 0] * 2)
    proba = np.array([0.9, 0.1, 0.9, 0.1] * 2)
    sensitive = pd.Series(["female"] * 4 + ["male"] * 4, name="gender")
    result = equalized_odds(y, proba, sensitive, 0.30, config)
    assert result["equalized_odds_tpr_diff"] == pytest.approx(0.0)
    assert result["equalized_odds_fpr_diff"] == pytest.approx(0.0)
    assert result["passes"]


# --- WARNING 발생 조건 ----------------------------------------------------

def test_di_warning_below_threshold(config, caplog):
    """DI < 0.8 이면 WARNING."""
    proba = np.array([0.1] * 4 + [0.9] * 6 + [0.1] * 8 + [0.9] * 2)
    with caplog.at_level(logging.WARNING, logger="src.fairness.metrics"):
        disparate_impact(proba, _gender(10, 10), 0.30, "female", config)
    assert any("DI Ratio" in r.message for r in caplog.records)
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_di_warning_above_threshold(config, caplog):
    """DI > 1.25 도 WARNING (역차별)."""
    proba = np.array([0.1] * 9 + [0.9], dtype=float)
    proba = np.concatenate([proba, np.array([0.1] * 4 + [0.9] * 6)])
    with caplog.at_level(logging.WARNING, logger="src.fairness.metrics"):
        result = disparate_impact(proba, _gender(10, 10), 0.30, "female", config)
    assert result["di_ratio"] > config["thresholds"]["di_ratio_max"]
    assert any("DI Ratio" in r.message for r in caplog.records)


def test_di_no_warning_within_range(config, caplog):
    proba = np.array([0.1] * 5 + [0.9] * 5 + [0.1] * 5 + [0.9] * 5)
    with caplog.at_level(logging.WARNING, logger="src.fairness.metrics"):
        disparate_impact(proba, _gender(10, 10), 0.30, "female", config)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_eo_warning_above_limit(config, caplog):
    """EO 차이 > 0.10 이면 WARNING."""
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0] * 2)
    proba = np.array([0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.1, 0.1,
                      0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    sensitive = pd.Series(["female"] * 8 + ["male"] * 8, name="gender")
    with caplog.at_level(logging.WARNING, logger="src.fairness.metrics"):
        equalized_odds(y, proba, sensitive, 0.30, config)
    messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("TPR" in m for m in messages) and any("FPR" in m for m in messages)


# --- evaluate 통합 --------------------------------------------------------

def test_evaluate_covers_all_sensitive_attributes(config):
    rng = np.random.default_rng(0)
    n = 400
    frame = pd.DataFrame({
        "gender": rng.choice(["male", "female"], n),
        "age_group": rng.choice(["20-34", "35-54", "55+"], n),
    })
    proba = rng.random(n)
    y = (rng.random(n) < 0.1).astype(int)
    result = evaluate(y, proba, frame, config)
    for attribute in config["fairness"]["sensitive_attributes"]:
        assert "disparate_impact" in result[attribute]
        assert "equalized_odds" in result[attribute]
    assert 0.0 <= result["overall_approval_rate"] <= 1.0


# --- Mitigation -----------------------------------------------------------

@pytest.fixture(scope="module")
def biased_setup(config):
    """한 집단의 피처만 체계적으로 낮춘 편향 데이터와 모델."""
    rng = np.random.default_rng(7)
    n = 4000
    gender = rng.choice(["male", "female"], n)
    signal = rng.normal(size=n)
    y = (signal + rng.normal(scale=0.7, size=n) > 1.0).astype(int)
    # 측정 편향: female 의 관측 신호만 낮춘다(실제 위험은 동일).
    observed = signal - np.where(gender == "female", 0.7, 0.0)

    frame = pd.DataFrame({"f0": observed, "f1": rng.normal(size=n)})
    frame["gender"] = gender
    frame["SeriousDlqin2yrs"] = y

    cols = ["f0", "f1"]
    split = n // 2
    train, rest = frame.iloc[:split], frame.iloc[split:]
    valid, test = rest.iloc[: len(rest) // 2], rest.iloc[len(rest) // 2:]
    model = LGBMClassifier(n_estimators=80, num_leaves=15, class_weight="balanced",
                           random_state=42, n_jobs=2, verbose=-1)
    model.fit(train[cols], train["SeriousDlqin2yrs"])
    return model, cols, valid.reset_index(drop=True), test.reset_index(drop=True)


def test_mitigation_improves_equalized_odds(config, biased_setup):
    """ADR-004: ThresholdOptimizer(equalized_odds) 적용 후 EO 차이가 줄어야 한다."""
    model, cols, valid, test = biased_setup
    optimizer = fit_threshold_optimizer(
        model, cols, valid, valid["SeriousDlqin2yrs"], valid["gender"].rename("gender"), config
    )
    predictions = optimizer.predict(test[cols], sensitive_features=test["gender"].to_numpy(), random_state=42)
    proba = model.predict_proba(test[cols])[:, 1]
    result = compare(
        test["SeriousDlqin2yrs"].to_numpy(), proba, predictions,
        test["gender"].rename("gender"), "female", config,
    )
    before_eo = max(result["before"]["equalized_odds"]["equalized_odds_tpr_diff"],
                    result["before"]["equalized_odds"]["equalized_odds_fpr_diff"])
    after_eo = max(result["after"]["equalized_odds"]["equalized_odds_tpr_diff"],
                   result["after"]["equalized_odds"]["equalized_odds_fpr_diff"])
    assert after_eo < before_eo, f"EO 차이가 줄지 않았다: {before_eo:.4f} -> {after_eo:.4f}"


def test_compare_structure(config, biased_setup):
    """전/후 비교 결과의 구조와 delta 계산."""
    model, cols, valid, test = biased_setup
    optimizer = fit_threshold_optimizer(
        model, cols, valid, valid["SeriousDlqin2yrs"], valid["gender"].rename("gender"), config
    )
    predictions = optimizer.predict(test[cols], sensitive_features=test["gender"].to_numpy(), random_state=42)
    proba = model.predict_proba(test[cols])[:, 1]
    result = compare(
        test["SeriousDlqin2yrs"].to_numpy(), proba, predictions,
        test["gender"].rename("gender"), "female", config,
    )
    for phase in ("before", "after"):
        assert {"disparate_impact", "equalized_odds", "approval_rate", "auc"} <= set(result[phase])
    assert result["delta"]["di_ratio"] == pytest.approx(
        result["after"]["disparate_impact"]["di_ratio"]
        - result["before"]["disparate_impact"]["di_ratio"]
    )
    assert result["criteria"]["di_min"] == config["thresholds"]["di_ratio_min"]


def test_adjusted_thresholds_cover_all_groups(config, biased_setup):
    """집단마다 조정된 임계값 규칙이 나와야 한다."""
    model, cols, valid, _ = biased_setup
    optimizer = fit_threshold_optimizer(
        model, cols, valid, valid["SeriousDlqin2yrs"], valid["gender"].rename("gender"), config
    )
    thresholds = adjusted_thresholds(optimizer)
    assert set(thresholds) == {"male", "female"}
    for rule in thresholds.values():
        assert rule, "임계값 규칙이 비어 있다"
