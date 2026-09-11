"""SHAP 설명 모듈 검증 (SPEC §3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from src.config import load_config, load_yaml
from src.data.preprocessor import feature_columns
from src.explainer.explainer import (
    CreditExplainer,
    _has_final_consonant,
    _subject_particle,
    base_value_scalar,
    shap_values_2d,
)


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def fitted(config):
    """combined 피처 구성의 소형 모델과 참조 데이터."""
    cols = feature_columns(config, "combined")
    rng = np.random.default_rng(0)
    n = 1500
    data = pd.DataFrame(rng.normal(size=(n, len(cols))), columns=cols)
    # 대안 피처는 0~1 범위, 연체 횟수는 0 이상으로 현실적인 값을 준다.
    for col in config["features"]["alternative"]:
        data[col] = rng.uniform(0.55, 1.0, n)
    for col in config["preprocess"]["missing_indicator_columns"]:
        data[col] = rng.integers(0, 4, n).astype(float)
        data[f"{col}{config['preprocess']['missing_flag_suffix']}"] = rng.integers(0, 2, n)
    data["MonthlyIncome"] = rng.lognormal(8.4, 0.6, n)
    data["age"] = rng.integers(21, 85, n)

    y = pd.Series(
        (0.6 - data["telecom_payment_rate"] + 0.15 * data["NumberOfTimes90DaysLate"]
         + rng.normal(0, 0.2, n) > 0).astype(int)
    )
    model = LGBMClassifier(n_estimators=60, num_leaves=15, random_state=42,
                           n_jobs=2, verbose=-1).fit(data[cols], y)

    reference = data.copy()
    reference["age_group"] = np.where(
        reference["age"] < 35, "20-34", np.where(reference["age"] < 55, "35-54", "55+")
    )
    return model, cols, reference, data, y


@pytest.fixture(scope="module")
def explainer(config, fitted):
    model, cols, reference, _, _ = fitted
    return CreditExplainer(model, cols, reference, config)


# --- SHAP 값 형태 ---------------------------------------------------------

def test_shap_values_shape(explainer, fitted, config):
    """SHAP 값은 (n_samples, n_features) 2D 여야 한다."""
    _, cols, _, data, _ = fitted
    sample = data.head(50)
    values, base = explainer.explain(sample)
    assert values.shape == (50, len(cols))
    assert isinstance(base, float)


def test_shap_values_2d_handles_3d_input():
    """3D (n, features, classes) 입력이 오면 양성 클래스를 고른다."""
    class Fake:
        values = np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2)
    out = shap_values_2d(Fake())
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, Fake.values[..., 1])


def test_shap_values_2d_handles_2d_and_1d():
    class Fake2D:
        values = np.zeros((4, 5))
    assert shap_values_2d(Fake2D()).shape == (4, 5)

    class Fake1D:
        values = np.zeros(5)
    assert shap_values_2d(Fake1D()).shape == (1, 5)


def test_shap_values_2d_rejects_bad_shape():
    class Bad:
        values = np.zeros((2, 2, 2, 2))
    with pytest.raises(ValueError):
        shap_values_2d(Bad())


def test_base_value_scalar_handles_class_axis():
    class Fake:
        base_values = np.array([[0.1, 0.9], [0.1, 0.9]])
    assert base_value_scalar(Fake()) == pytest.approx(0.9)


def test_shap_additivity(explainer, fitted):
    """SHAP 값 합 + base ≈ 모델의 로짓 출력 (가법성)."""
    model, cols, _, data, _ = fitted
    sample = data.head(20)
    values, base = explainer.explain(sample)
    raw = model.predict_proba(sample[cols])[:, 1]
    logit = np.log(raw / (1 - raw))
    np.testing.assert_allclose(values.sum(axis=1) + base, logit, atol=1e-4)


# --- 거절 사유 ------------------------------------------------------------

REQUIRED_FIELDS = ("feature", "current_value", "shap_contribution", "explanation")


def test_rejection_reasons_returns_five(explainer, fitted, config):
    """SPEC §3: 거절 사유 Top 5."""
    _, _, reference, _, _ = fitted
    row = reference.iloc[0]
    reasons = explainer.rejection_reasons(row, row["age_group"])
    assert len(reasons) == config["explainer"]["top_rejection_reasons"] == 5


def test_rejection_reasons_required_fields(explainer, fitted):
    """SPEC §3 이 요구하는 4개 필드 + 한글명/참조값."""
    _, _, reference, _, _ = fitted
    row = reference.iloc[1]
    for reason in explainer.rejection_reasons(row, row["age_group"]):
        for field in REQUIRED_FIELDS:
            assert field in reason, f"{field} 누락"
        assert "feature_name_kr" in reason and "reference_value" in reason
        assert isinstance(reason["explanation"], str) and reason["explanation"]
        assert "기여도" in reason["explanation"]


def test_rejection_reasons_are_ranked_and_adverse(explainer, fitted):
    """rank 는 1부터, 기여도는 내림차순이 아니라 '불리한 순'(오름차순)이어야 한다."""
    _, _, reference, _, _ = fitted
    row = reference.iloc[2]
    reasons = explainer.rejection_reasons(row, row["age_group"])
    assert [r["rank"] for r in reasons] == [1, 2, 3, 4, 5]
    contributions = [r["shap_contribution"] for r in reasons]
    assert contributions == sorted(contributions), "가장 불리한 요인이 1순위여야 한다"
    # 부호 규약: shap_contribution = -shap_value_default
    for reason in reasons:
        assert reason["shap_contribution"] == pytest.approx(-reason["shap_value_default"], abs=1e-4)


def test_display_row_changes_values_not_ranking(explainer, fitted, config):
    """display_row 는 표시값만 바꾸고 SHAP 순위는 바꾸지 않아야 한다.

    회귀 방지: 초기 구현은 역스케일된 값을 모델에 넣어 SHAP 을 계산해
    워터폴과 거절 사유가 서로 다른 피처를 가리켰다.
    """
    _, cols, reference, _, _ = fitted
    suffix = config["preprocess"]["missing_flag_suffix"]
    row = reference.iloc[4]
    display = row.copy()
    for col in cols:
        if col.endswith(suffix):
            continue                                    # 결측 플래그는 0/1 표지이지 표시값이 아니다
        display[col] = float(row[col]) * 3.0 + 7.0      # 표시용으로만 크게 왜곡

    plain = explainer.rejection_reasons(row, row["age_group"])
    with_display = explainer.rejection_reasons(row, row["age_group"], display_row=display)

    assert [r["feature"] for r in plain] == [r["feature"] for r in with_display]
    assert [r["shap_contribution"] for r in plain] == [r["shap_contribution"] for r in with_display]
    for base, shown in zip(plain, with_display):
        if base["feature"].endswith(suffix) or "기록이 없습니다" in base["explanation"]:
            # 플래그 자체이거나 플래그가 선 연체 값: 표시값과 무관하게 같은 값으로 표기한다
            assert shown["current_value"] == base["current_value"]
            continue
        assert shown["current_value"] == pytest.approx(base["current_value"] * 3.0 + 7.0)


def test_sentinel_with_missing_flag_is_described_as_no_record(explainer, fitted, config):
    """센티넬(-1) + 결측 플래그 1 인 연체 값은 '기록 없음'으로 서술해야 한다.

    회귀 방지: API 는 null 을 그대로 넘겨 '기록 없음' 분기를 탔지만, 배치 경로(run_explain)는
    역스케일한 센티넬을 넘겨 "30-59일 연체 횟수가 -1회로, ... 대비 적습니다" 가 출력됐다.
    """
    _, cols, reference, _, _ = fitted
    pre = config["preprocess"]
    column = pre["missing_indicator_columns"][0]
    row = reference.iloc[0].copy()
    row[column] = float(pre["missing_sentinel"])
    row[f"{column}{pre['missing_flag_suffix']}"] = 1

    reasons = explainer.contributing_factors(row, row["age_group"], top_n=len(cols))
    target = next(r for r in reasons if r["feature"] == column)

    assert "기록이 없습니다" in target["explanation"]
    assert "-1회" not in target["explanation"]


def test_rejection_reasons_respect_top_n(explainer, fitted):
    _, _, reference, _, _ = fitted
    row = reference.iloc[3]
    assert len(explainer.rejection_reasons(row, row["age_group"], top_n=3)) == 3


# --- reference_value ------------------------------------------------------

def test_reference_value_matches_age_group_mean(explainer, fitted):
    """reference_value 는 동일 연령대 평균이어야 한다."""
    _, _, reference, _, _ = fitted
    for group in ("20-34", "35-54", "55+"):
        expected = reference.loc[reference["age_group"] == group, "telecom_payment_rate"].mean()
        assert explainer.reference_value("telecom_payment_rate", group) == pytest.approx(expected)


def test_reference_value_differs_between_groups(explainer):
    values = {g: explainer.reference_value("age", g) for g in ("20-34", "35-54", "55+")}
    assert values["20-34"] < values["35-54"] < values["55+"]


def test_reference_value_falls_back_to_overall(explainer, fitted):
    """없는 집단이면 전체 평균으로 대체한다."""
    _, _, reference, _, _ = fitted
    overall = reference["telecom_payment_rate"].mean()
    assert explainer.reference_value("telecom_payment_rate", "없는집단") == pytest.approx(overall)


# --- 자연어 설명 · 부호 방향 ----------------------------------------------

def test_direction_consistent_when_value_is_unfavourable(explainer):
    """값이 평균보다 낮고(불리) 기여도도 음수(불리)면 일치."""
    text, consistent = explainer._describe("telecom_payment_rate", 0.78, 0.92, -0.15)
    assert consistent
    assert "낮습니다" in text
    assert "78%" in text and "92%" in text
    assert "(기여도: -0.15)" in text


def test_direction_inconsistent_is_flagged(explainer):
    """값은 유리한데 기여도가 불리하면 불일치로 표시하고 문장에 단서를 붙인다."""
    text, consistent = explainer._describe("telecom_payment_rate", 0.95, 0.90, -0.10)
    assert not consistent
    assert "조합에 따른 영향" in text


def test_direction_for_lower_is_better_feature(explainer):
    """연체 횟수는 낮을수록 좋다. 평균보다 많은데 기여도 음수면 일치."""
    _, consistent = explainer._describe("NumberOfTimes90DaysLate", 3.0, 0.4, -0.22)
    assert consistent
    _, inconsistent = explainer._describe("NumberOfTimes90DaysLate", 0.0, 0.4, -0.22)
    assert not inconsistent


def test_unit_templates(explainer):
    """단위별 문장 템플릿이 분기돼야 한다."""
    percent, _ = explainer._describe("telecom_payment_rate", 0.78, 0.92, -0.15)
    count, _ = explainer._describe("NumberOfTimes90DaysLate", 3.0, 0.4, -0.22)
    amount, _ = explainer._describe("MonthlyIncome", 2_500_000.0, 5_400_000.0, -0.08)
    years, _ = explainer._describe("age", 27.0, 52.0, -0.03)
    assert "%" in percent
    assert "회" in count and ("많습니다" in count or "적습니다" in count)
    assert "2,500,000" in amount and "5,400,000" in amount
    assert "세" in years


def test_korean_particle_selection():
    """받침 유무에 따라 이/가 를 고른다."""
    assert _has_final_consonant("납부율") and _subject_particle("납부율") == "이"
    assert not _has_final_consonant("빈도") and _subject_particle("빈도") == "가"


def test_feature_name_mapping_covers_all_features(config):
    """모든 학습 피처에 한글명이 있어야 한다."""
    names = load_yaml(config["explainer"]["feature_names_file"])["features"]
    for group in ("traditional_only", "alternative_only", "combined"):
        for feature in feature_columns(config, group):
            assert feature in names, f"{feature} 한글명 매핑 누락"
            assert "kr" in names[feature] and "unit" in names[feature]
            assert "higher_is_better" in names[feature]


# --- 전역 중요도 ----------------------------------------------------------

def test_global_importance(explainer, fitted, config):
    _, cols, _, data, _ = fitted
    importance = explainer.global_importance(data.head(300))
    assert len(importance) == len(cols)
    assert importance["mean_abs_shap"].is_monotonic_decreasing
    assert set(importance["feature_type"]) <= {"alternative", "traditional"}
    # 타겟을 만들 때 쓴 telecom_payment_rate 가 상위권이어야 한다.
    assert "telecom_payment_rate" in importance.head(3)["feature"].tolist()
