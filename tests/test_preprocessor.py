"""전처리 · 분할 검증 (SPEC §1 전처리 및 분할, §7 제약 사항)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data import simulator as sim
from src.data.preprocessor import (
    FEATURE_GROUPS,
    Preprocessor,
    feature_columns,
    prepare,
    split_data,
)
from tests.test_simulator import _fake_gmsc


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def raw(config) -> pd.DataFrame:
    return sim.simulate(_fake_gmsc(20_000), config)


@pytest.fixture(scope="module")
def prepared(raw, config):
    return prepare(raw, config, save=False)


# --- 분할 -----------------------------------------------------------------

def test_split_ratio(raw, config):
    """70:15:15 분할 비율 (config 값 기준)."""
    parts = split_data(raw, config)
    total = len(raw)
    for name in ("train", "valid", "test"):
        share = len(parts[name][0]) / total
        assert abs(share - config["split"][name]) < 0.005, f"{name} 비율 {share:.4f}"
    assert sum(len(parts[n][0]) for n in parts) == total


def test_split_is_disjoint(raw, config):
    """분할 간 행이 겹치지 않아야 한다."""
    parts = split_data(raw, config)
    indices = [set(parts[n][0].index) for n in ("train", "valid", "test")]
    assert not indices[0] & indices[1]
    assert not indices[0] & indices[2]
    assert not indices[1] & indices[2]


def test_class_ratio_preserved(raw, config):
    """stratify=y 이므로 각 분할의 양성 비율이 전체와 같아야 한다."""
    target = config["data"]["target_column"]
    overall = raw[target].mean()
    parts = split_data(raw, config)
    for name in ("train", "valid", "test"):
        assert abs(parts[name][1].mean() - overall) < 0.005, f"{name} 양성률 이탈"


def test_split_is_reproducible(raw, config):
    """random_state 가 config 에 고정돼 있으므로 재현 가능해야 한다."""
    first = split_data(raw, config)
    second = split_data(raw, config)
    for name in ("train", "valid", "test"):
        assert list(first[name][0].index) == list(second[name][0].index)


# --- 누수 방지 -------------------------------------------------------------

def test_scaler_fitted_on_train_only(prepared, config):
    """train 은 평균 0 / 표준편차 1 에 맞춰지고, valid·test 는 정확히 0 이 아니어야 한다."""
    processed, pre = prepared
    train_x = processed["train"][0]
    numeric = pre.numeric_columns
    assert np.allclose(train_x[numeric].mean().to_numpy(), 0, atol=1e-6)
    assert np.allclose(train_x[numeric].std(ddof=0).to_numpy(), 1, atol=1e-6)

    # valid/test 에 다시 fit 했다면 평균이 0 이 됐을 것이다. 그렇지 않아야 한다.
    for name in ("valid", "test"):
        means = processed[name][0][numeric].mean().to_numpy()
        assert not np.allclose(means, 0, atol=1e-6), f"{name} 에 스케일러를 다시 fit 한 것으로 보인다"


def test_median_comes_from_train(raw, config):
    """중앙값 대체값은 train 에서만 계산돼야 한다."""
    parts = split_data(raw, config)
    pre = Preprocessor(config).fit(parts["train"][0])
    for col, value in pre.medians.items():
        assert value == pytest.approx(float(parts["train"][0][col].median()))


def test_transform_requires_fit(config):
    """fit 없이 transform 하면 실패해야 한다."""
    with pytest.raises(RuntimeError):
        Preprocessor(config).transform(pd.DataFrame({"age": [30]}))


def test_target_not_in_features(prepared, config):
    """전처리 결과에 타겟이 섞여 들어가면 안 된다(라벨 누수)."""
    processed, _ = prepared
    for name in ("train", "valid", "test"):
        assert config["data"]["target_column"] not in processed[name][0].columns


def test_feature_columns_identical_across_splits(prepared):
    """세 분할의 피처 구성이 완전히 같아야 한다."""
    processed, _ = prepared
    cols = list(processed["train"][0].columns)
    assert list(processed["valid"][0].columns) == cols
    assert list(processed["test"][0].columns) == cols


# --- 결측 처리 -------------------------------------------------------------

def test_delinquency_missing_preserved_as_flag(prepared, config):
    """연체 3컬럼 결측은 중앙값 대체가 아니라 결측 플래그로 보존돼야 한다."""
    processed, _ = prepared
    suffix = config["preprocess"]["missing_flag_suffix"]
    for col in config["preprocess"]["missing_indicator_columns"]:
        flag = f"{col}{suffix}"
        assert flag in processed["train"][0].columns, f"{flag} 컬럼이 없다"
        assert set(processed["train"][0][flag].unique()) <= {0, 1}
        assert processed["train"][0][flag].sum() > 0, "결측 플래그가 전부 0이다"


def test_delinquency_uses_sentinel_not_median(raw, config):
    """센티넬 값이 적용되고, 중앙값 대체 대상에 연체 컬럼이 들어가 있지 않아야 한다."""
    pre = Preprocessor(config)
    sentinel = config["preprocess"]["missing_sentinel"]
    marked = pre._add_missing_indicators(raw)
    for col in config["preprocess"]["missing_indicator_columns"]:
        assert not marked[col].isna().any()
        assert (marked[col] == sentinel).sum() > 0
        assert col not in config["preprocess"]["median_impute_columns"]
    # 중앙값 대체 대상에도 포함되지 않는다.
    parts = split_data(raw, config)
    fitted = Preprocessor(config).fit(parts["train"][0])
    for col in config["preprocess"]["missing_indicator_columns"]:
        assert col not in fitted.medians


def test_flag_matches_delinquency_thin_filer_types(prepared, config):
    """ADR-010: 결측 플래그는 연체결측 유형(missing_only/both)과 일치해야 한다.

    씬파일러 전체와 일치하지 않는다. history_only 유형은 연체 컬럼이 관측되기 때문이다.
    """
    processed, _ = prepared
    suffix = config["preprocess"]["missing_flag_suffix"]
    train_x = processed["train"][0]
    flag = f"{config['preprocess']['missing_indicator_columns'][0]}{suffix}"
    expected = train_x["thin_filer_type"].isin(["missing_only", "both"]).astype(int)
    pd.testing.assert_series_equal(train_x[flag].astype(int), expected, check_names=False)
    # 씬파일러 전체보다 적어야 한다(history_only 만큼 차이).
    assert train_x[flag].sum() < train_x["is_thin_filer"].sum()


def test_no_missing_values_after_preprocess(prepared):
    """전처리 후 결측이 남아 있으면 안 된다."""
    processed, _ = prepared
    for name in ("train", "valid", "test"):
        assert not processed[name][0].isna().any().any(), f"{name} 에 결측이 남았다"


# --- 인코딩 ---------------------------------------------------------------

def test_protected_attributes_preserved_raw(prepared, config):
    """ADR-008: 보호 속성은 One-Hot 하지 않고 원본 값 그대로 유지한다."""
    processed, pre = prepared
    train_x = processed["train"][0]
    for col in config["features"]["protected"]:
        assert col in train_x.columns, f"{col} 이 사라졌다 (공정성 평가에 필요)"
        # One-Hot 컬럼이 만들어지지 않아야 한다.
        assert not [c for c in train_x.columns if c.startswith(f"{col}_")]
    assert set(train_x["gender"].unique()) <= set(config["fairness"]["groups"]["gender"])
    assert set(train_x["age_group"].unique()) <= set(config["fairness"]["groups"]["age_group"])
    assert pre.categories == {}, "보호 속성이 One-Hot 대상에 남아 있다"


def test_protected_and_segment_excluded_from_features(config):
    """ADR-008 / ADR-009: 보호 속성과 세그먼트 컬럼은 어떤 피처 그룹에도 없어야 한다."""
    excluded = set(config["features"]["protected"]) | set(config["features"]["segment"])
    for group in FEATURE_GROUPS:
        cols = set(feature_columns(config, group))
        assert not (cols & excluded), f"{group} 에 제외 대상이 포함됨: {cols & excluded}"


def test_feature_groups_composition(config, prepared):
    """피처 그룹 구성: traditional 은 결측 플래그 포함, combined 는 두 그룹의 합집합."""
    processed, _ = prepared
    train_x = processed["train"][0]
    traditional = feature_columns(config, "traditional_only")
    alternative = feature_columns(config, "alternative_only")
    combined = feature_columns(config, "combined")

    assert set(combined) == set(traditional) | set(alternative)
    assert not set(traditional) & set(alternative)
    suffix = config["preprocess"]["missing_flag_suffix"]
    assert [c for c in traditional if c.endswith(suffix)], "결측 플래그가 전통 그룹에 없다"
    for group in FEATURE_GROUPS:
        for col in feature_columns(config, group):
            assert col in train_x.columns, f"{col} 이 전처리 결과에 없다"


def test_unknown_feature_group_raises(config):
    with pytest.raises(ValueError):
        feature_columns(config, "nope")


def test_numeric_passthrough_not_scaled(prepared, config):
    """is_thin_filer 는 0/1, credit_history_months 는 원래 개월수 범위를 유지해야 한다."""
    processed, _ = prepared
    train_x = processed["train"][0]
    assert set(train_x["is_thin_filer"].unique()) <= {0, 1}
    months = train_x["credit_history_months"]
    assert months.min() >= config["simulator"]["credit_history"]["thin_range"][0]
    assert months.max() <= config["simulator"]["credit_history"]["normal_range"][1]


def test_save_and_load_roundtrip(prepared, tmp_path):
    """전처리기를 저장했다가 불러와도 같은 결과가 나와야 한다."""
    processed, pre = prepared
    path = pre.save(tmp_path / "preprocessor.joblib")
    restored = Preprocessor.load(path)
    assert restored.medians == pre.medians
    assert restored.feature_names == pre.feature_names
