"""loader.load_gmsc() 검증.

원본 GMSC(data/raw/cs-training.csv)가 있어야 실행된다. 없으면 skip.
"""

from __future__ import annotations

import pytest

from src.data.loader import MIN_AGE, load_config, load_gmsc, resolve_path

RAW_ROWS = 150_000          # 원본 행 수
EXPECTED_ROWS = 149_999     # ADR-007 age<18 제거 후


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def df(config):
    path = resolve_path(config["paths"]["data_raw"]) / config["data"]["gmsc_file"]
    if not path.exists():
        pytest.skip(f"원본 데이터 없음: {path} (README 의 다운로드 절차 참조)")
    return load_gmsc(config)


def test_row_count(df):
    """age 필터 후 행 수. 원본 150,000행에서 age<18 인 1건이 빠진다."""
    assert len(df) == EXPECTED_ROWS
    assert RAW_ROWS - len(df) == 1


def test_target_positive_rate(df, config):
    """부도율은 약 6.7% (스펙 기준). 6.0~7.5% 범위를 벗어나면 데이터가 바뀐 것."""
    target = config["data"]["target_column"]
    assert target in df.columns
    assert set(df[target].unique()) == {0, 1}
    rate = df[target].mean()
    assert 0.060 <= rate <= 0.075, f"양성 비율 이상: {rate:.4f}"
    assert round(rate, 5) == pytest.approx(0.06684, abs=1e-4)


def test_missing_columns(df):
    """결측은 MonthlyIncome 과 NumberOfDependents 두 컬럼에만 존재한다."""
    na = df.isna().sum()
    with_missing = set(na[na > 0].index)
    assert with_missing == {"MonthlyIncome", "NumberOfDependents"}
    assert na["MonthlyIncome"] / len(df) == pytest.approx(0.198, abs=0.01)
    assert na["NumberOfDependents"] / len(df) == pytest.approx(0.026, abs=0.01)


def test_age_minimum(df):
    """ADR-007: age 최솟값은 MIN_AGE 이상이어야 한다."""
    assert df["age"].min() >= MIN_AGE
    assert (df["age"] < MIN_AGE).sum() == 0


def test_index_column_removed(df):
    """원본의 이름 없는 행 번호 컬럼이 제거됐는지."""
    assert not any(str(c).startswith("Unnamed") for c in df.columns)
    assert df.shape[1] == 11


def test_paths_are_relative(config):
    """config 의 경로 설정은 모두 상대경로여야 한다."""
    for key, value in config["paths"].items():
        assert not str(value).startswith("/"), f"paths.{key} 가 절대경로다: {value}"
